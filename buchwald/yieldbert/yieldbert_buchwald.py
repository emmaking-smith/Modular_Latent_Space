'''
Module for Predicting Yield-Bert on Buchwald-Hartwig Data.
'''

import argparse
import pandas as pd
from collections import Counter
import pkg_resources
from rdkit import Chem
import torch
import numpy as np
import os

from rxnfp.models import SmilesClassificationModel
from rxn_yields.data import generate_buchwald_hartwig_rxns

model_path =  pkg_resources.resource_filename(
                "rxnfp",
                f"models/transformers/bert_pretrained"
)

model_args = {
     'num_train_epochs': 15, 'overwrite_output_dir': True,
    'learning_rate': 0.00009659, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": False,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.7987 }
}

yield_bert = SmilesClassificationModel("bert", model_path, num_labels=1,
                                       args=model_args, use_cuda=torch.cuda.is_available())


def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def remove_nans(df):
    df.loc[pd.isna(df['aryl_halide_smiles']) == True, ['aryl_halide_smiles']] = ''
    df.loc[pd.isna(df['ligand_smiles']) == True, ['ligand_smiles']] = ''
    df.loc[pd.isna(df['base_smiles']) == True, ['base_smiles']] = ''
    df.loc[pd.isna(df['additive_smiles']) == True, ['additive_smiles']] = ''
    return df

def find_halide(df_row):
    return canonicalize_smiles(df_row['aryl_halide_smiles'])

def find_ligand(df_row):
    return canonicalize_smiles(df_row['ligand_smiles'])

def find_base(df_row):
    return canonicalize_smiles(df_row['base_smiles'])

def find_additive(df_row):
    return canonicalize_smiles(df_row['additive_smiles'])

def find_mol_ranking(df):
    '''
    Finding most common electrophiles and nucleophiles used in rxns.
    '''
    halides = [find_halide(df.iloc[x]) for x in list(df.index)]
    ligands = [find_ligand(df.iloc[x]) for x in list(df.index)]
    bases = [find_base(df.iloc[x]) for x in list(df.index)]
    additives = [find_additive(df.iloc[x]) for x in list(df.index)]

    ranked_halides = [Counter(halides).most_common()[x][0] for x in range(len(Counter(halides).most_common()))]
    ranked_ligands = [Counter(ligands).most_common()[x][0] for x in range(len(Counter(ligands).most_common()))]
    ranked_bases = [Counter(bases).most_common()[x][0] for x in range(len(Counter(bases).most_common()))]
    ranked_additives = [Counter(additives).most_common()[x][0] for x in range(len(Counter(additives).most_common()))]

    return ranked_halides, ranked_ligands, ranked_bases, ranked_additives

def split_on_molecule(df, canon_smiles, reagent_type):
    '''
    Given a reactant to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''

    assert reagent_type in ['ligand', 'base']

    excludes_mol_df = df.copy()
    includes_mol_df = df.copy()

    includes_mol_idxs = []
    for i in list(df.index):
        if reagent_type == 'ligand':
            reactants = find_ligand(df.iloc[i])
        else:
            reactants = find_base(df.iloc[i])

        if canon_smiles == reactants:
            includes_mol_idxs.append(i)

    excludes_react_idxs = list(set(list(df.index)) - set(includes_mol_idxs))

    includes_mol_df = includes_mol_df.drop(excludes_react_idxs)
    excludes_mol_df = excludes_mol_df.drop(includes_mol_idxs)

    return excludes_mol_df, includes_mol_df

def split_on_molecules(df, list_of_canon_smiles, reagent_type):
    '''
    Given a reactant to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''

    assert reagent_type in ['halide', 'additive']

    excludes_mol_df = df.copy()
    includes_mol_df = df.copy()

    includes_mol_idxs = []
    for i in list(df.index):
        if reagent_type == 'halide':
            reactants = find_halide(df.iloc[i])
        else:
            reactants = find_additive(df.iloc[i])

        for smiles in list_of_canon_smiles:
            if smiles == '':
                if smiles == reactants:
                    includes_mol_idxs.append(i)
            else:
                if smiles in reactants:
                    includes_mol_idxs.append(i)

    excludes_react_idxs = list(set(list(df.index)) - set(includes_mol_idxs))

    includes_mol_df = includes_mol_df.drop(excludes_react_idxs)
    excludes_mol_df = excludes_mol_df.drop(includes_mol_idxs)

    return excludes_mol_df, includes_mol_df

class YieldBert_Setup():
    '''
    Make a reaction string.
    '''
    def __init__(self, df):
        self.df = df

    def find_product(self, df_row):
        product = df_row['product_smiles']
        if pd.isna(product) == True:
            canon_product = find_halide(df_row)
        else:
            canon_product = canonicalize_smiles(product)
        return canon_product

    def find_all_reactants_reagents_product(self, row_number):
        halide = find_halide(self.df.iloc[row_number])
        ligand = find_ligand(self.df.iloc[row_number])
        base = find_base(self.df.iloc[row_number])
        additive = find_additive(self.df.iloc[row_number])
        product = self.find_product(self.df.iloc[row_number])
        return halide, ligand, base, additive, product

    def make_rxn_string(self, row_number):
        halide, ligand, base, additive, product = self.find_all_reactants_reagents_product(row_number)
        reactants = halide + '.' + ligand + '.' + base + '.' + additive
        if reactants[-1] == '.':
            reactants = reactants[:-1]
        return reactants + '>>' + product

    def make_yieldbert_label(self, row_number):
        return self.df.loc[row_number, 'yield']

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_buchwald', type=str,
                        help='The paths to the Buchwald csv file.',
                        default='../doyle_buchwald_data.csv')
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--split', type=str, help='Options are: halide, ligand, base, additive.')
    parser.add_argument('--test_mol_idx', type=int, help='What molecule to leave out for testing.')
    return parser.parse_args()

def main():
    # Init arguments.
    args = init_args()
    path_to_buchwald = args.path_to_buchwald
    save_path = args.save_path
    split = args.split
    test_mol_idx = args.test_mol_idx

    assert split in ['halide', 'ligand', 'base', 'additive']

    # Load in data.
    buchwald_rxns = pd.read_csv(path_to_buchwald)
    buchwald_rxns = remove_nans(buchwald_rxns)

    if split in ['ligand', 'base']:
        dataframe_splits = split_on_molecule

    else:
        dataframe_splits = split_on_molecules

    ranked_halides, ranked_ligands, ranked_bases, ranked_additives = find_mol_ranking(buchwald_rxns)
    if split == 'halide':
        test_smiles = ranked_halides[test_mol_idx * 4: (test_mol_idx + 1) * 4]
    elif split == 'ligand':
        test_smiles = ranked_ligands[test_mol_idx]
    elif split == 'base':
        test_smiles = ranked_bases[test_mol_idx]
    else:
        test_smiles = ranked_additives[test_mol_idx * 6: (test_mol_idx + 1) * 6]

    train_df, test_df = dataframe_splits(buchwald_rxns, test_smiles, split)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Creating the train_df.
    train_text = []
    train_labels = []
    for i in range(len(train_df)):
        train_text.append(YieldBert_Setup(train_df).make_rxn_string(i))
        train_labels.append(YieldBert_Setup(train_df).make_yieldbert_label(i))

    yieldbert_train = pd.DataFrame({'text' : train_text, 'labels' : train_labels})

    mean = np.mean(yieldbert_train['labels'])
    std = np.std(yieldbert_train['labels'])
    yieldbert_train['labels'] = (yieldbert_train['labels'] - mean) / std

    # Creating the val_df.
    test_text = []
    test_labels = []
    for i in range(len(test_df)):
        test_text.append(YieldBert_Setup(test_df).make_rxn_string(i))
        test_labels.append(YieldBert_Setup(test_df).make_yieldbert_label(i))

    yieldbert_test = pd.DataFrame({'text': test_text, 'labels': test_labels})

    # print(test_text[0])
    # print(test_labels[0])
    yieldbert_test['labels'] = (yieldbert_test['labels'] - mean) / std

    yield_bert.train_model(yieldbert_train, output_dir=save_path, eval_df=yieldbert_test)

    last_checkpoint = ''
    checkpoints = os.listdir(save_path)
    for check in checkpoints:
        if "epoch-15" in check:
            last_checkpoint = check

    model_path = os.path.join(save_path, last_checkpoint)

    trained_yield_bert = SmilesClassificationModel('bert', model_path,
                                                   num_labels=1, args={
            "regression": True
        }, use_cuda=torch.cuda.is_available())

    yield_predicted = trained_yield_bert.predict(yieldbert_test.text.values)[0] * std + mean

    yield_true = yieldbert_test.labels.values * std + mean

    overall_losses = []
    for pred, true in zip(yield_predicted, yield_true):
        print(f"predicted: {pred:.1f} | true: {true:.1f}")
        loss = np.abs(pred - true)
        print(f"loss: {loss}")
        overall_losses.append(loss)
        print()

    print(f'Overall Mean MAE: {np.mean(overall_losses)}')
    # print(f'Overall Median MAE: {np.median(overall_losses)}')
    # chemist_losses = []
    # for loss in overall_losses:
    #     if loss <= 10:
    #         chemist_losses.append(0)
    #     else:
    #         chemist_losses.append(loss)
    # print(f'Overall Mean Chemist MAE: {np.mean(chemist_losses)}')

if __name__ == '__main__':
    main()
