import warnings
warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
import shutil
import json
import sys
import time
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess


URL = "https://github.com/AllenNeuralDynamics/aind-capsule-ephys-curation"
VERSION = "0.1.0"


curation_params = dict(
        isi_violations_ratio_threshold=0.5,
        presence_ratio_threshold=0.8,
        amplitude_cutoff_threshold=0.1,
    )


job_kwargs = {
    'n_jobs': -1,
    'chunk_duration': '1s',
    'progress_bar': True
}

data_folder = Path("../data/")
scratc_folder = Path("../scratch")
results_folder = Path("../results/")


if __name__ == "__main__":
    data_processes_folder = results_folder / "data_processes_curation"
    data_processes_folder.mkdir(exist_ok=True, parents=True)
    
    si.set_global_job_kwargs(**job_kwargs)

    ####### CURATION ########
    print("\nCURATION")
    curation_notes = ""

    t_curation_start_all = time.perf_counter()

    # curation query
    isi_violations_ratio_thr = curation_params["isi_violations_ratio_threshold"]
    presence_ratio_thr = curation_params["presence_ratio_threshold"]
    amplitude_cutoff_thr = curation_params["amplitude_cutoff_threshold"]

    curation_query = f"isi_violations_ratio < {isi_violations_ratio_thr} and presence_ratio > {presence_ratio_thr} and amplitude_cutoff < {amplitude_cutoff_thr}"
    print(f"Curation query: {curation_query}")
    curation_notes += f"Curation query: {curation_query}\n"

    # check if test
    if (data_folder / "postprocessing_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_folder = data_folder / "postprocessing_output_test" / "postprocessed"
        
        curation_query = f"isi_violations_ratio < {isi_violations_ratio_thr} and amplitude_cutoff < {amplitude_cutoff_thr}"
        del curation_params["presence_ratio_threshold"]
    else:
        postprocessed_folder = data_folder / "postprocessed"

    if not postprocessed_folder.is_dir():
        print("'postprocessed' folder not found. Exiting")
        sys.exit(1)

    postprocessed_folders = [p for p in postprocessed_folder.iterdir() if "_sorting" not in p.name]
    for postprocessed_folder in postprocessed_folders:
        datetime_start_curation = datetime.now()
        t_curation_start = time.perf_counter()
        recording_name = postprocessed_folder.name
        curation_output_process_json = data_processes_folder / f"curation_{recording_name}.json"

        print(f"Curating recording: {recording_name}")

        we = si.load_waveforms(postprocessed_folder, with_recording=False)

        # get quality metrics
        qm = we.load_extension("quality_metrics").get_data()
        qm_curated = qm.query(curation_query)
        curated_unit_ids = qm_curated.index.values

        # flag units as good/bad depending on QC selection
        qc_quality = [True if unit in curated_unit_ids else False for unit in we.sorting.unit_ids]
        sorting_precurated = we.sorting
        sorting_precurated.set_property("default_qc", qc_quality)
        sorting_precurated.save(folder=results_folder / "sorting_precurated" / recording_name)
        n_units = int(len(sorting_precurated.unit_ids))
        n_passing = int(np.sum(qc_quality))
        print(f"\t{np.sum(qc_quality)}/{len(sorting_precurated.unit_ids)} passing default QC.\n")
        curation_notes += f"{np.sum(qc_quality)}/{len(sorting_precurated.unit_ids)} passing default QC.\n"
        t_curation_end = time.perf_counter()
        elapsed_time_curation = np.round(t_curation_end - t_curation_start, 2)

        # save params in output
        curation_params["recording_name"] = recording_name
        
        curation_outputs = dict(
            total_units=n_units,
            passing_qc=n_passing,
            failing_qc=n_units - n_passing
        )
        curation_process = DataProcess(
                name="Ephys curation",
                version=VERSION, # either release or git commit
                start_date_time=datetime_start_curation,
                end_date_time=datetime_start_curation + timedelta(seconds=np.floor(elapsed_time_curation)),
                input_location=str(data_folder),
                output_location=str(results_folder),
                code_url=URL,
                parameters=curation_params,
                outputs=curation_outputs,
                notes=curation_notes
            )
        with open(curation_output_process_json, "w") as f:
            f.write(curation_process.json(indent=3))

    t_curation_end_all = time.perf_counter()
    elapsed_time_curation_all = np.round(t_curation_end_all - t_curation_start_all, 2)
    print(f"CURATION time: {elapsed_time_curation_all}s")
