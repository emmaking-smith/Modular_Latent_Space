# Transfer Learning for a Modular and Generalizable Chemistry Latent Space
Code for associated paper as described in King-Smith *et al.*

## Getting Your Bearings
Please use the following modules as a guide when setting up finetuning on your own tasks. Examples of regression (MAE) tasks can be found in the Toxicity, Buchwald, and Suzuki directories. An example of a classification task can be found in the Fragrance directory.

### MPNN:
Modules and code associated with the base message passing neural network (MPNN).
* The core message passing neural network used to pretrain on crystal structure data and for each finetuning task.
* The pretrained weights and biases for the modular latent space.

### Toxicity:
Modules and code associated with LD50 predictions of small molecule compounds.
* Data from the Therapeutic Data Commons [TDC](https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50) (3 x pickle files).
* Toxicity data on non-therapeutic small molecules. (1 x pickle file).
* Baseline machine learning predictions.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Tox.

### Suzuki:
Modules and code associated with LD50 predictions of small molecule compounds.

**Note: For [Yield-BERT](https://rxn4chemistry.github.io/rxn_yields/) and [GraphRXN](https://github.com/jidushanbojue/GraphRXN/tree/master) comparisons, please create separate conda environments as directed by the associated documentation!** 
* Historical literature Suzuki coupling data from USPTO (2 x txt files).
* Baseline machine learning predictions.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Yield on Suzuki data.
* Associated code for running comparisons with Yield-BERT and GraphRXN.

### Buchwald:
Modules and code associated with the Buchwald-Hartwig yield predictions.

**Note: For [Yield-BERT](https://rxn4chemistry.github.io/rxn_yields/) and [GraphRXN](https://github.com/jidushanbojue/GraphRXN/tree/master) comparisons, please create separate conda environments as directed by the associated documentation!** 
* The high throughput experimentation Buchwald-Hartwig data from Ahneman *et al.* (csv).
* Baseline machine learning predictions.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Yield on Buchwald-Hartwig data.
* Associated code for running comparisons with Yield-BERT and GraphRXN.

### Fragrance:
Modules and code associated with multiclass and multilabel odor class predictions of small molecules.
* Pyrfume data (1 x pickle file).
* AI Crowd enantiomer data (1 x pickle file).
* Baseline machine learning predictions on Pyrfume data.
* Baseline machine learning predictions on enantiomeric pairs.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Olfaction on Pyrfume data.
* Module to run Crystal-Olfaction on enantiomeric pairs.

## Dependencies:
Run on python 3.7.
* numpy==1.21.5
* pandas==1.3.5
* sklearn==1.0.2
* rdkit==2020.09.1
* torch==1.10.0+cu113
* [networkx==1.11](https://pypi.org/project/networkx/)






