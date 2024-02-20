import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
import argparse
import json
import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc


# AIND
from aind_data_schema.core.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-curation"
VERSION = "1.0"

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

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")


if __name__ == "__main__":
    args = parser.parse_args()

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    N_JOBS = int(N_JOBS_CO) if N_JOBS_CO is not None else N_JOBS

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_curation"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    curation_params = processing_params["curation"]

    ####### CURATION ########
    print("\nCURATION")
    curation_notes = ""

    t_curation_start_all = time.perf_counter()

    # curation query
    isi_violations_ratio_thr = curation_params["isi_violations_ratio_threshold"]
    presence_ratio_thr = curation_params["presence_ratio_threshold"]
    amplitude_cutoff_thr = curation_params["amplitude_cutoff_threshold"]

    # check if test
    if (data_folder / "postprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_folder = data_folder / "postprocessing_pipeline_output_test"

        curation_query = (
            f"isi_violations_ratio < {isi_violations_ratio_thr} and amplitude_cutoff < {amplitude_cutoff_thr}"
        )
        del curation_params["presence_ratio_threshold"]
    else:
        curation_query = f"isi_violations_ratio < {isi_violations_ratio_thr} and presence_ratio > {presence_ratio_thr} and amplitude_cutoff < {amplitude_cutoff_thr}"
        postprocessed_folder = data_folder

    print(f"Curation query: {curation_query}")
    curation_notes += f"Curation query: {curation_query}\n"

    postprocessed_folders = [
        p for p in postprocessed_folder.iterdir() if "postprocessed_" in p.name and "-sorting" not in p.name
    ]
    for postprocessed_folder in postprocessed_folders:
        datetime_start_curation = datetime.now()
        t_curation_start = time.perf_counter()
        recording_name = ("_").join(postprocessed_folder.name.split("_")[1:])
        curation_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"

        try:
            we = si.load_waveforms(postprocessed_folder, with_recording=False)
            print(f"Curating recording: {recording_name}")
        except:
            print(f"Spike sorting failed on {recording_name}. Skipping curation")
            # create an mock result file (needed for pipeline)
            mock_qc = np.array([], dtype=bool)
            np.save(results_folder / f"qc_{recording_name}.npy", mock_qc)
            continue

        # get quality metrics
        qm = we.load_extension("quality_metrics").get_data()
        qm_curated = qm.query(curation_query)
        curated_unit_ids = qm_curated.index.values

        # flag units as good/bad depending on QC selection
        default_qc = np.array([True if unit in curated_unit_ids else False for unit in we.sorting.unit_ids])
        n_passing = int(np.sum(default_qc))
        n_units = len(we.unit_ids)
        print(f"\t{n_passing}/{n_units} passing default QC.\n")
        curation_notes += f"{n_passing}/{n_units} passing default QC.\n"
        # save flags to results folder
        np.save(results_folder / f"qc_{recording_name}.npy", default_qc)
        t_curation_end = time.perf_counter()
        elapsed_time_curation = np.round(t_curation_end - t_curation_start, 2)

        # save params in output
        curation_params["recording_name"] = recording_name

        curation_outputs = dict(total_units=n_units, passing_qc=n_passing, failing_qc=n_units - n_passing)
        curation_process = DataProcess(
            name="Ephys curation",
            software_version=VERSION,  # either release or git commit
            start_date_time=datetime_start_curation,
            end_date_time=datetime_start_curation + timedelta(seconds=np.floor(elapsed_time_curation)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=curation_params,
            outputs=curation_outputs,
            notes=curation_notes,
        )
        with open(curation_output_process_json, "w") as f:
            f.write(curation_process.model_dump_json(indent=3))

    t_curation_end_all = time.perf_counter()
    elapsed_time_curation_all = np.round(t_curation_end_all - t_curation_start_all, 2)
    print(f"CURATION time: {elapsed_time_curation_all}s")
