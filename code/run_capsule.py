import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import time
import logging
from datetime import datetime, timedelta
import pandas as pd

# SPIKEINTERFACE
import spikeinterface as si
from spikeinterface.core.core_tools import check_json
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as scur

# AIND
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName

from huggingface_hub.utils import logging as hf_logging
hf_logging.set_verbosity_error()

try:
    from aind_log_utils import log
    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-curation"
VERSION = "2.0"

data_folder = Path("../data/")
scratch_folder = Path("../scratch")
results_folder = Path("../results/")

# Define argument parser
parser = argparse.ArgumentParser(description="Curate ecephys data")

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default="-1", help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")



if __name__ == "__main__":
    ####### CURATION ########
    curation_notes = ""
    t_curation_start_all = time.perf_counter()

    args = parser.parse_args()

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS = args.params

    # Use CO_CPUS/SLURM_CPUS_ON_NODE env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("SLURM_CPUS_ON_NODE")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else N_JOBS

    if PARAMS is not None:
        try:
            # try to parse the JSON string first to avoid file name too long error
            curation_params = json.loads(PARAMS)
        except json.JSONDecodeError:
            if Path(PARAMS).is_file():
                with open(PARAMS, "r") as f:
                    curation_params = json.load(f)
            else:
                raise ValueError(f"Invalid parameters: {PARAMS} is not a valid JSON string or file path")
    else:
        with open("params.json", "r") as f:
            curation_params = json.load(f)

    data_process_prefix = "data_process_curation"

    job_kwargs = curation_params.pop("job_kwargs")
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    ecephys_sorted_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir() and "ecephys" in p.name or "behavior" in p.name and "sorted" in p.name
    ]

    # look for subject and data_description JSON files
    subject_id = "undefined"
    session_name = "undefined"
    for f in data_folder.iterdir():
        # the file name is {recording_name}_subject.json
        if "subject.json" in f.name:
            with open(f, "r") as file:
                subject_id = json.load(file)["subject_id"]
        # the file name is {recording_name}_data_description.json
        if "data_description.json" in f.name:
            with open(f, "r") as file:
                session_name = json.load(file)["name"]

    if HAVE_AIND_LOG_UTILS:
        log.setup_logging(
            "Curate Ecephys",
            subject_id=subject_id,
            asset_name=session_name,
        )
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info("\nCURATION")

    pipeline_mode = True
    if len(ecephys_sorted_folders) > 0:
        # capsule mode
        assert len(ecephys_sorted_folders) == 1, "Attach one sorted asset at a time"
        ecephys_sorted_folder = ecephys_sorted_folders[0]
        postprocessed_base_folder = ecephys_sorted_folder / "postprocessed"
        pipeline_mode = False
    elif (data_folder / "postprocessing_pipeline_output_test").is_dir():
        logging.info("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_base_folder = data_folder / "postprocessing_pipeline_output_test"
    else:
        postprocessed_base_folder = data_folder

    if pipeline_mode:
        postprocessed_folders = [
            p for p in postprocessed_base_folder.iterdir() if "postprocessed_" in p.name
        ]
    else:
        postprocessed_folders = [
            p for p in postprocessed_base_folder.iterdir() if "postprocessed-sorting" not in p.name and p.is_dir()
        ]
    for postprocessed_folder in postprocessed_folders:
        datetime_start_curation = datetime.now()
        t_curation_start = time.perf_counter()
        if pipeline_mode:
            recording_name = ("_").join(postprocessed_folder.name.split("_")[1:])
        else:
            recording_name = postprocessed_folder.name
        if recording_name.endswith(".zarr"):
            recording_name = recording_name[:recording_name.find(".zarr")]
        curation_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"

        try:
            analyzer = si.load(postprocessed_folder)
            logging.info(f"Curating recording: {recording_name}")
        except Exception as e:
            logging.info(f"Spike sorting failed on {recording_name}. Skipping curation")
            # create an mock result file (needed for pipeline)
            mock_qc = np.array([], dtype=bool)
            np.save(results_folder / f"qc_{recording_name}.npy", mock_qc)
            mock_df = pd.DataFrame()
            mock_df.to_csv(results_folder / f"unit_classifier_{recording_name}.csv")
            continue

        n_units = int(len(analyzer.unit_ids))

        # pass/fail default QC
        qc_thresholds = curation_params["qc_thresholds"]
        logging.info(f"Curation thresholds: {qc_thresholds}")
        curation_notes += f"Curation thresholds: {qc_thresholds}\n"

        qm = analyzer.get_extension("quality_metrics").get_data()
        default_qc_labels = scur.threshold_metrics_label_units(
            qm,
            thresholds=qc_thresholds,
            pass_label=True,
            fail_label=False,
            column_name="default_qc"
        )
        n_passing_qc = int(np.sum(default_qc_labels["default_qc"]))
        logging.info(f"\tPassing default QC: {n_passing_qc} / {n_units}")
        curation_notes += f"Passing default QC: {n_passing_qc}/{n_units}"
        all_labels = [default_qc_labels]

        # UnitRefine (noise/mua/sua)

        # patch for wrong template metrics dtypes.
        # not sure why this happens, but casting to float doesn't hurt
        template_metrics_ext = analyzer.get_extension("template_metrics")
        template_metrics_ext.data["metrics"] = template_metrics_ext.data["metrics"].replace("<NA>","NaN").astype("float32")

        noise_neural_classifier = curation_params.get(
            "noise_neural_classifier",
            "SpikeInterface/UnitRefine_noise_neural_classifier"
        )
        sua_mua_classifier = curation_params.get(
            "sua_mua_classifier",
            "SpikeInterface/UnitRefine_sua_mua_classifier"
        )
        logging.info(f"Applying UnitRefine with: {noise_neural_classifier} -- {sua_mua_classifier}")

        unitrefine_labels = scur.unitrefine_label_units(
            analyzer,
            noise_neural_classifier=noise_neural_classifier,
            sua_mua_classifier=sua_mua_classifier
        )
        prediction = unitrefine_labels["unitrefine_label"]

        n_unitrefine_sua = int(np.sum(prediction == "sua"))
        n_unitrefine_mua = int(np.sum(prediction == "mua"))
        n_unitrefine_noise = int(np.sum(prediction == "noise"))

        logging.info(f"\tUnitRefine Noise: {n_unitrefine_noise} / {n_units}")
        logging.info(f"\tUnitRefine SUA: {n_unitrefine_sua} / {n_units}")
        logging.info(f"\tUnitRefine MUA: {n_unitrefine_mua} / {n_units}")

        curation_notes += f"UnitRefine Noise: {n_unitrefine_noise} / {n_units}\n"
        curation_notes += f"UnitRefine SUA: {n_unitrefine_sua} / {n_units}\n"
        curation_notes += f"UnitRefine MUA: {n_unitrefine_mua} / {n_units}\n"
        all_labels.append(unitrefine_labels)

        # Bombcell
        bombcell_params = curation_params.get("bombcell")
        try:
            bombcell_labels = scur.bombcell_label_units(analyzer, thresholds=bombcell_params)

            n_bombcell_sua = int(np.sum(prediction == "good"))
            n_bombcell_mua = int(np.sum(prediction == "mua"))
            n_bombcell_noise = int(np.sum(prediction == "noise"))
            n_bombcell_non_somatic = int(np.sum(prediction == "noise"))

            logging.info(f"\tBombcell Noise: {n_bombcell_noise} / {n_units}")
            logging.info(f"\tBombcell SUA: {n_bombcell_sua} / {n_units}")
            logging.info(f"\tBombcell MUA: {n_bombcell_mua} / {n_units}")
            logging.info(f"\tBombcell NON-SOMA: {n_bombcell_non_somatic} / {n_units}")

            curation_notes += f"Bombcell Noise: {n_bombcell_noise} / {n_units}\n"
            curation_notes += f"Bombcell SUA: {n_bombcell_sua} / {n_units}\n"
            curation_notes += f"Bombcell MUA: {n_bombcell_mua} / {n_units}\n"
            curation_notes += f"Bombcell NON-SOMA: {n_bombcell_non_somatic} / {n_units}\n"
            all_labels.append(bombcell_labels)
        except Exception as e:
            bombcell_labels = None
            logging.info(f"Failed to apply bombcell labeling. Error:\n{e}")

        all_labels_df = pd.concat(all_labels, axis=1)
        all_labels_df.to_csv(results_folder / f"unit_labels_{recording_name}.csv", index=False)

        # Apply auto-merging
        slay_params = curation_params.get("slay")
        potential_merges = scur.compute_merge_unit_groups(
            analyzer,
            preset="slay",
            steps_params=slay_params
        )
        n_slay_merges = len(potential_merges)
        logging.info(f"\tSLAy found {len(potential_merges)} potential merges")
        curation_notes += f"SLAy found {len(potential_merges)} potential merges\n"

        with open(results_folder / f"unit_merges_{recording_name}.json", mode="w") as f:
            json.dump(check_json(potential_merges), f)
        
        t_curation_end = time.perf_counter()
        elapsed_time_curation = np.round(t_curation_end - t_curation_start, 2)

        # save params in output
        curation_params["recording_name"] = recording_name

        curation_outputs = dict(
            total_units=n_units, 
            passing_qc=n_passing_qc,
            failing_qc=n_units - n_passing_qc,
            unitrefine_noise=n_unitrefine_noise,
            unitrefine_sua=n_unitrefine_sua,
            unitrefine_mua=n_unitrefine_mua,
            slay_merges=n_slay_merges
        )

        if bombcell_labels is not None:
            curation_outputs.update(
                dict(
                    bombcell_noise=n_bombcell_noise,
                    bombcell_sua=n_bombcell_sua,
                    bombcell_mua=n_bombcell_mua,
                    bombcell_non_somatic=n_bombcell_non_somatic,
                )
            )

        if pipeline_mode:
            curation_process = DataProcess(
                process_type=ProcessName.EPHYS_CURATION,
                stage=ProcessStage.PROCESSING,
                name="Ephys curation",
                experimenters=["Alessio Buccino"],
                code=Code(
                    url=URL,
                    version=VERSION, # either release or git commit
                    parameters=curation_params
                ),
                start_date_time=datetime_start_curation,
                end_date_time=datetime_start_curation + timedelta(seconds=np.floor(elapsed_time_curation)),
                output_path=str(results_folder),
                output_parameters=curation_outputs,
                notes=curation_notes,
            )
            with open(curation_output_process_json, "w") as f:
                f.write(curation_process.model_dump_json(indent=3))

    t_curation_end_all = time.perf_counter()
    elapsed_time_curation_all = np.round(t_curation_end_all - t_curation_start_all, 2)
    logging.info(f"CURATION time: {elapsed_time_curation_all}s")
