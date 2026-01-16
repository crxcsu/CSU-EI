<div style="text-align: center;">
    <img src="logo.png" width="200px">
</div>

## CSU-EI

This is the code repo for the paper *Contrastively Spectral-structural Unification for Electron–Ionization Mass Spectrometry*. We developed a method named Contrastively Spectral-structural Unification for Electron Ionization mass spectrometry (CSU-EI)

### Package required: 
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html). 

By using the [`requirements/environment.yaml`](requirements/environment.yaml)file, it will install all the required packages.

    git clone https://github.com/csu-jialong/CSU-EI.git
    cd CSU-EI
    conda env create -f requirements/environment.yaml
    conda activate csu-ei

## Interacting with CSU-EI

In [`Interacting_with_CSUEI.ipynb`](Interacting_with_CSUEI.ipynb) file.

Or [`https://huggingface.co/spaces`](https://huggingface.co/spaces)

## Codebase guide
configs folder:
- `config.json`: model_config, dataset_config, training_config

data folder:
- `7_test`: 7 extra test set molecules
- `example_smiles_db`: 110 molecules incuding 10 extra test molecules
- `extra_10_test`: 10 extra test set molecules
  
fig folder: Some visualization pictures

loss_function: loss function we used in training stage

model folder: 
- `tokenizer-smiles-roberta`: dict for tokenizer
- `CSU-EI.py`: model logic
- `tokenizer.py`: RobertaTokenizer

model_weight: CSU-EI model weights file

requirements:
- `environment.yaml`: requied packages

utils:
- `data_preprocessing.py`: function for data preprocessing
- `plot_eims.py`: plot EI-MS spectra
- `NetCDF.py`: read .cdf files


## Information of maintainers
- zmzhang@csu.edu.cn
- 232311034@csu.edn.cn