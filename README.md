# Transfer Learning for a Modular and Generalizable Chemistry Latent Space
Code for associated paper as described in King-Smith *et al.*

## Getting Your Bearings
Please use the following modules as a guide when setting up finetuning on your own tasks. Examples of regression (MAE) tasks can be found in the Toxicity, Buchwald, and Suzuki directories. An example of a classification task can be found in the Fragrance directory.

## MPNN
Modules and code associated with the base message passing neural network (MPNN).
* The core message passing neural network used to pretrain on crystal structure data and for each finetuning task.
* The pretrained weights and biases for the modular latent space.

## Toxicity
Modules and code associated with LD50 predictions of small molecule compounds.
* Data from the Therapeutic Data Commons [TDC](https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50) (3 x pickle files).
* Toxicity data on non-therapeutic small molecules. (1 x pickle file).
* Baseline machine learning predictions.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Tox.

To run a prediction with Crystal-Tox, run the following command:

```
python toxicity/predict_toxicity.py --path_to_toxicity_data {PATH TO TRAINING DATA} {PATH TO VALIDATION DATA} {PATH TO TESTING DATA}
                                    --save_path {PATH TO SAVE DIRECTORY}
```

The training, validation, and testing data should be pickle files of the form observed in `toxicity/data/new_compounds_for_testing_TDC.pickle`. Default training, validation, and testing data is from TDC. 

Predictions will be saved out as a pickle file as `preds.pickle` in your `{PATH TO SAVE DIRECTORY}`. The trained model dict will be saved out as `model` in your `{PATH TO SAVE DIRECTORY}`.

## Suzuki
Modules and code associated with LD50 predictions of small molecule compounds.

>[!Note] 
>For [Yield-BERT](https://rxn4chemistry.github.io/rxn_yields/) and [GraphRXN](https://github.com/jidushanbojue/GraphRXN/tree/master) comparisons, >please create separate conda environments as directed by the associated documentation!
* Historical literature Suzuki coupling data from USPTO (2 x txt files).
* Baseline machine learning predictions.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Yield on Suzuki data.
* Associated code for running comparisons with Yield-BERT and GraphRXN.

To run a prediction with Crystal-Yield with Suzuki USPTO data, run the following command:

```
python suzuki/suzuki_yield_prediction.py --paths_to_parsed_suzuki_data {PATH TO DATA 1} {PATH TO DATA 2}
                                         --split {nucleophile / electrophile}
                                         --save_path {PATH TO SAVE DIRECTORY}
```

The data should be texts files of the form observed in `suzuki/suzuki_from_arom_USPTO_parsed_het.txt`. Default text data is from processed Suzuki USPTO reactions. The split refers to which unseen molecule(s) will be used for testing; the options are `nucleophile` or `electrophile` (case sensitive). 

Predictions will be saved out as a pickle file as `preds.pickle` in your `{PATH TO SAVE DIRECTORY}`. The trained model dict will be saved out as `model` in your `{PATH TO SAVE DIRECTORY}`.

## Buchwald
Modules and code associated with the Buchwald-Hartwig yield predictions.

**Note: For [Yield-BERT](https://rxn4chemistry.github.io/rxn_yields/) and [GraphRXN](https://github.com/jidushanbojue/GraphRXN/tree/master) comparisons, please create separate conda environments as directed by the associated documentation!** 
* The high throughput experimentation (HTE) Buchwald-Hartwig data from Ahneman *et al.* (csv).
* Baseline machine learning predictions.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Yield on Buchwald-Hartwig data.
* Associated code for running comparisons with Yield-BERT and GraphRXN.

To run a prediction with Crystal-Yield with HTE-style data, run the following command:

```
python buchwald/buchwald_yield_prediction.py --path_to_buchwald_data {PATH TO CSV FILE}
                                             --split {halide / base / ligand / additive}
                                             --test_mol_idx {INTEGER FROM 0-3 if split == halide, ligand, additive / INTEGER FROM 0-2 if split == base}
                                             --save_path {PATH TO SAVE DIRECTORY}
```

The data should be a csv file of the form observed in `buchwald/doyle_buchwald_data.csv`. Default csv data is HTE data from Ahneman *et al.* The `split` and `test_mol_idx` point to which unseen molecule(s) will be used for testing; the options for the `split` are `halide`, `base`, `ligand`, or `additive` (case sensitive). Options for the `test_mol_idx` are 0-3 for `halide`, `ligand`, and `additive` splits and 0-2 for `base` splits.

Predictions will be saved out as a pickle file as `preds.pickle` in your `{PATH TO SAVE DIRECTORY}`. The trained model dict will be saved out as `model` in your `{PATH TO SAVE DIRECTORY}`.

## Fragrance
Modules and code associated with multiclass and multilabel odor class predictions of small molecules.
* Pyrfume data (1 x pickle file).
* AI Crowd enantiomer data (1 x pickle file).
* Baseline machine learning predictions on Pyrfume data.
* Baseline machine learning predictions on enantiomeric pairs.
* Dataset cleanup, MPNN setup, and preparation for pytorch dataloaders.
* Module to run Crystal-Olfaction on Pyrfume data.
* Module to run Crystal-Olfaction on enantiomeric pairs.

To run a prediction with Crystal-Olfaction with Pyrfume-style data, run the following command:

```
python fragrance/predict_fragrance.py --path_to_fragrance_data {PATH TO PYRFUME-STYLE PICKLE FILE}
                                      --fold {INTEGER FROM 0-4}
                                      --save_path {PATH TO SAVE DIRECTORY}
```

The data should be a pickle file of the form observed in `fragrance/data/pyrfume_enatiomer_pairs_removed.pickle`. Default pickle data is the Pyrfume data. The `fold` argument referrs to which fold in the 5-fold cross validation should be left out as the testing set. Options for `fold` are 0-4.

Predictions will be saved out as a pickle file as `preds.pickle` in your `{PATH TO SAVE DIRECTORY}`. The trained model dict will be saved out as `model` in your `{PATH TO SAVE DIRECTORY}`.

## Dependencies
Run on python 3.7.
* numpy==1.21.5
* pandas==1.3.5
* sklearn==1.0.2
* rdkit==2020.09.1
* torch==1.10.0+cu113
* [networkx==1.11](https://pypi.org/project/networkx/)






