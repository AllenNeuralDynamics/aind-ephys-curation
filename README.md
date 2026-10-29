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

Parameters are read from a `params.json` file that must be in the root folder of the capsule (next to `code/`, i.e. `../params.json` from `code/`), with the step parameters in the `curation` section.

A full list of parameters can be found in the  `aind-ephys-pipeline` repository, and are defined in this [schema](https://github.com/AllenNeuralDynamics/aind-ephys-pipeline/blob/1.5.0/pipeline/default_params_schema.json).

### Output

The output of this capsule is the following:

- `results/qc_{recording_name}.npy` file, containing the boolean labels from the automatic curation
- `results/data_process_curation_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

