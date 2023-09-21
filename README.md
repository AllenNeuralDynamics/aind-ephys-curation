# Curation for AIND ephys pipeline
## aind-ephys-curation


### Description

This capsule is designed to curate spike sorted data for the AIND pipeline.

It uses a quality metrics based *recipe* to flag units as passing or failing default quality control (QC).

The recipe is based on the following quality metrics:

- isi violation ratio < 0.5
- presence ratio > 0.8
- amplitude cutoff < 0.1


### Inputs

The `data/` folder must include the output of the [aind-ephys-postprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-postprocessing), including the `postprocessed_{recording_name}` folder.

### Parameters

The `code/run` script takes no arguments.

A list of curation thresholds used for curation can be found at the top of the `code/run_capsule.py` script:

```python
curation_params = dict(
    isi_violations_ratio_threshold=0.5,
    presence_ratio_threshold=0.8,
    amplitude_cutoff_threshold=0.1,
)
```

### Output

The output of this capsule is the following:

- `results/curated_{recording_name}` folder, containing the spike sorted data, saved by SpikeInterface, with the additional `default_QC` property
- `results/data_process_curation_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

