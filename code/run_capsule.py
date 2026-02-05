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
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as scur

# AIND
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName

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
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("SLURM_CPUS_ON_NODE") or os.getenv("SLURM_CPUS_PER_TASK")
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

        # pass/fail default QC
        curation_query = curation_params["query"]
        logging.info(f"Curation query: {curation_query}")
        curation_notes += f"Curation query: {curation_query}\n"

        qm = analyzer.get_extension("quality_metrics").get_data()
        qm_curated = qm.query(curation_query)
        curated_unit_ids = qm_curated.index.values

        # flag units as good/bad depending on QC selection
        default_qc = np.array([True if unit in curated_unit_ids else False for unit in analyzer.sorting.unit_ids])
        n_passing = int(np.sum(default_qc))
        n_units = len(analyzer.unit_ids)
        logging.info(f"\tPassing default QC: {n_passing} / {n_units}")
        curation_notes += f"Passing default QC: {n_passing}/{n_units}"
        # save flags to results folder
        np.save(results_folder / f"qc_{recording_name}.npy", default_qc)

        # estimate unit labels (noise/mua/sua)

        # patch for wrong template metrics dtypes.
        # not sure why this happens, but casting to float doesn't hurt
        template_metrics_ext = analyzer.get_extension("template_metrics")
        template_metrics_ext.data["metrics"] = template_metrics_ext.data["metrics"].replace("<NA>","NaN").astype("float32")

        # 1. apply the noise/neural classification and remove noise
        noise_neural_classifier = curation_params.get(
            "noise_neural_classifier",
            "SpikeInterface/UnitRefine_noise_neural_classifier"
        )
        logging.info(f"Applying noise-neural classifier from {noise_neural_classifier}")
        noise_neuron_labels = scur.auto_label_units(
            sorting_analyzer=analyzer,
            repo_id=noise_neural_classifier,
            trust_model=True,
        )
        noise_units = noise_neuron_labels[noise_neuron_labels['prediction'] == 'noise']

        # 2. apply the sua/mua classification and aggregate results
        if len(analyzer.unit_ids) > len(noise_units):
            sua_mua_classifier = curation_params.get(
                "sua_mua_classifier",
                "SpikeInterface/UnitRefine_sua_mua_classifier"
            )
            logging.info(f"Applying sua-mua classifier from {sua_mua_classifier}")

            analyzer_neural = analyzer.remove_units(noise_units.index)
            sua_mua_labels = scur.auto_label_units(
                sorting_analyzer=analyzer_neural,
                repo_id=sua_mua_classifier,
                trust_model=True,
            )
            all_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
        else:
            all_labels = noise_units

        all_labels = all_labels.rename(columns={"prediction": "decoder_label", "probability": "decoder_probability"})
        prediction = all_labels["decoder_label"]

        n_sua = int(np.sum(prediction == "sua"))
        n_mua = int(np.sum(prediction == "mua"))
        n_noise = int(np.sum(prediction == "noise"))
        n_units = int(len(analyzer.unit_ids))

        logging.info(f"\tNoise: {n_noise} / {n_units}")
        logging.info(f"\tSUA: {n_sua} / {n_units}")
        logging.info(f"\tMUA: {n_mua} / {n_units}")

        curation_notes += f"Noise: {n_noise} / {n_units}\n"
        curation_notes += f"SUA: {n_sua} / {n_units}\n"
        curation_notes += f"MUA: {n_mua} / {n_units}\n"

        all_labels.to_csv(results_folder / f"unit_classifier_{recording_name}.csv", index=False)
        
        t_curation_end = time.perf_counter()
        elapsed_time_curation = np.round(t_curation_end - t_curation_start, 2)

        # save params in output
        curation_params["recording_name"] = recording_name

        curation_outputs = dict(
            total_units=n_units, 
            passing_qc=n_passing, 
            failing_qc=n_units - n_passing, 
            noise_units=noise_units,
            neural_units=n_sua + n_mua,
            sua_unita=n_sua,
            mua_units=n_mua
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
