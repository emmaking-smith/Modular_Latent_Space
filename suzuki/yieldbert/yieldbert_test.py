'''
Module for Predicting Yield-Bert on Suzuki Data.
'''

import argparse
import pandas as pd
from collections import Counter
import pkg_resources
import re
from rdkit import Chem
from itertools import chain
import torch
import numpy as np
import os

from rxnfp.models import SmilesClassificationModel

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

def load_parsed_data(path, start_index=0):
    '''
    Returns a dictonary whose keys are the
    rxn ids and whose values are dictionaries containing
    that rxn's information.
    '''
    dict = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            dict[i + start_index] = eval(line)
    return dict

def find_reactants(rxn):
    '''
    Removes that atom numbering and orders reactants such that
    the nucleophile (boronic acid) is entry 0, and electrophile (aryl halide)
    is entry 1.
    '''
    reactants = [rxn[0][x] for x in range(len(rxn[0])) if type(rxn[0][x]) == str]
    nucleophile = 0
    electrophile = 0
    for reactant in reactants:
        if 'B(' in reactant:
            nucleophile = reactant
        else:
            electrophile = reactant
    assert nucleophile != 0 and electrophile != 0
    ordered_canon_reactants = [remove_atom_mapped_numbering(nucleophile), remove_atom_mapped_numbering(electrophile)]
    return ordered_canon_reactants

def remove_atom_mapped_numbering(smiles):
    smiles = re.sub(':[0-9]+', '', smiles)
    canonical_smiles = canonicalize_smiles(smiles)
    return canonical_smiles

def split_on_reactant(rxn_dict, canon_reactant_smiles):
    '''
    Given a reactant to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''
    excludes_reactant_dict = rxn_dict.copy()
    includes_reactant_dict = rxn_dict.copy()
    includes_reactant_idxs = []
    for key in list(rxn_dict.keys()):
        reactants = find_reactants(rxn_dict[key]['rx'])
        if canon_reactant_smiles in reactants:
            includes_reactant_idxs.append(key)
    excludes_react_idxs = list(set(list(rxn_dict.keys())) - set(includes_reactant_idxs))
    for key in includes_reactant_idxs:
        del excludes_reactant_dict[key]
    for key in excludes_react_idxs:
        del includes_reactant_dict[key]

    return excludes_reactant_dict, includes_reactant_dict

def split_on_multiple_electrophiles(rxn_dict, canon_electrophile_smiles_list):
    '''
    Splits into training / testing / validation sets for multiple electrophile smiles.
    '''
    excludes_reactant_dict = rxn_dict.copy()
    includes_reactant_dict = rxn_dict.copy()
    includes_reactant_idxs = []
    for key in list(rxn_dict.keys()):
        reactants = find_reactants(rxn_dict[key]['rx'])
        if len(list(set(canon_electrophile_smiles_list) & set(reactants))) > 0:
            includes_reactant_idxs.append(key)
    excludes_react_idxs = list(set(list(rxn_dict.keys())) - set(includes_reactant_idxs))
    for key in includes_reactant_idxs:
        del excludes_reactant_dict[key]
    for key in excludes_react_idxs:
        del includes_reactant_dict[key]

    return excludes_reactant_dict, includes_reactant_dict

def split_on_cat_lig(rxn_dict, canon_cat_lig_smiles):
    '''
    Given a catalyst or ligand to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''
    excludes_reactant_dict = rxn_dict.copy()
    includes_reactant_dict = rxn_dict.copy()
    includes_reactant_idxs = []
    for key in list(rxn_dict.keys()):
        catalyst, ligand = find_catalysts_ligands(rxn_dict[key])
        catalyst = reformat_ligs_and_cats([catalyst])
        catalyst = canonicalize_smiles(catalyst[0])

        ligand = clean_ligands([ligand])
        ligand = reformat_ligs_and_cats(ligand)
        ligand = canonicalize_smiles(ligand[0])

        cat_lig = [catalyst, ligand]

        if canon_cat_lig_smiles in cat_lig:
            includes_reactant_idxs.append(key)
    excludes_react_idxs = list(set(list(rxn_dict.keys())) - set(includes_reactant_idxs))
    for key in includes_reactant_idxs:
        del excludes_reactant_dict[key]
    for key in excludes_react_idxs:
        del includes_reactant_dict[key]

    return excludes_reactant_dict, includes_reactant_dict

def ranking_nucleophiles_and_electrophiles(rxn_dict):
    '''
    Finding most common electrophiles and nucleophiles used in rxns.
    '''
    nucleophiles = [find_reactants(rxn_dict[x]['rx'])[0] for x in list(rxn_dict.keys())]
    electrophiles = [find_reactants(rxn_dict[x]['rx'])[1] for x in list(rxn_dict.keys())]
    most_common_nucleophiles = Counter(nucleophiles).most_common()[0:2] # OB(O)c1ccccc1 most common nucleophile
    most_common_electrophiles = Counter(electrophiles).most_common()[0:60] # Brc1ccccn1 most common electrophile
    return [x[0] for x in most_common_nucleophiles], [x[0] for x in most_common_electrophiles]

def reformat_ligs_and_cats(cleaned_ligs_or_cats):
    reformatted = []
    for smiles_list in cleaned_ligs_or_cats:
        reformat = '.'.join(smiles_list)
        reformatted.append(reformat)
    return reformatted

def ranking_catalysts_and_ligands(rxn_dict):
    '''
    Finding most common catalysts. Note that catalysts do NOT have atom mapping to get rid of.
    '''
    # Catalysts
    catalysts = [find_catalysts_ligands(rxn_dict[x])[0] for x in list(rxn_dict.keys())]
    catalysts = reformat_ligs_and_cats(catalysts)
    catalysts = [canonicalize_smiles(x) for x in catalysts]
    most_common_catalysts = Counter(catalysts).most_common()[1:3]

    # Ligands
    ligands = [find_catalysts_ligands(rxn_dict[x])[1] for x in list(rxn_dict.keys())]
    ligands = clean_ligands(ligands)
    ligands = reformat_ligs_and_cats(ligands)
    ligands = [canonicalize_smiles(x) for x in ligands]
    most_common_ligands = Counter(ligands).most_common()[1:3] # most common ligand is no ligand at all.

    return [x[0] for x in most_common_catalysts], [x[0] for x in most_common_ligands]


def find_catalysts_ligands(rxn):
    '''
    Finding the catalysts and ligands used in each reaction.
    Note that some rxns may not use ligands (e.g. just the catalyst / a precatalyst was used).
    '''
    catalyst = rxn['catalysts']
    ligand = list(rxn['ligands'])
    return catalyst, ligand

def detuple(set_of_tuple):
    return list(chain.from_iterable(set_of_tuple))

def clean_ligands(ligand_list):
    cleaned_ligands = []
    for ligs in ligand_list:
        if len(ligs) == 0:
            cleaned_ligands.append(ligs)
        elif len(ligs) == 1:
            if type(list(ligs)[0]) == tuple:
                cleaned_ligands.append(detuple(ligs))
            else:
                cleaned_ligands.append(ligs)
        else:
            mixed_tuple_ligands = []
            for l in ligs:
                if type(l) != str:
                    mixed_tuple_ligands = mixed_tuple_ligands + list(l)
                else:
                    mixed_tuple_ligands.append(l)

            cleaned_ligands.append(mixed_tuple_ligands)
    return cleaned_ligands

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_to_parsed_suzuki_data', type=list, help='The paths to the parsed Suzuki text files.',
                        default=['suzuki_from_arom_USPTO_parsed_het.txt',
                                 'suzuki_from_arom_USPTO_parsed_homo.txt'])
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--split', type=str, help='Options are: nucleophile, electrophile, catalyst, ligand.')
    return parser.parse_args()

class YieldBert_Setup():
    def __init__(self):
        pass

    def clean_ligands(self, ligand_list):
        if len(ligand_list) == 0:
            return []
        elif len(ligand_list) == 1:
            if type(ligand_list[0]) == str:
                return ligand_list
            else:
                untupled_ligands = list(chain.from_iterable(ligand_list))
                return untupled_ligands
        else:
            cleaned_ligands = []
            for lig in ligand_list:
                if type(lig) == str:
                    cleaned_ligands.append(lig)
                else:
                    untupled_ligands = list(lig)
                    cleaned_ligands = cleaned_ligands + untupled_ligands
            return cleaned_ligands

    def reformat_ligs_and_cats(self, cleaned_ligs_or_cats):
        if len(cleaned_ligs_or_cats) == 1:
            return cleaned_ligs_or_cats[0]
        elif len(cleaned_ligs_or_cats) == 0:
            return ''
        else:
            reformat = '.'.join(cleaned_ligs_or_cats)
            return reformat

    def convert_dict_to_yieldbert_text(self, single_rxn):

        boronic_acid, halide = find_reactants(single_rxn['rx'])
        catalysts, ligands = find_catalysts_ligands(single_rxn)

        # Formating catalysts to be of 'mol1.mol2. ... ' instead of a list within a list.
        catalysts = self.reformat_ligs_and_cats(catalysts)
        catalysts = canonicalize_smiles(catalysts)

        # Cleaning the ligands to not be sets and tuples but only lists.
        ligands = self.clean_ligands(ligands)
        ligands = self.reformat_ligs_and_cats(ligands)

        ligands = canonicalize_smiles(ligands)

        product = self.find_product(single_rxn)

        # Convert to reaction string.
        reaction_string = self.make_reaction_string(boronic_acid, halide, catalysts, ligands, product)
        return reaction_string

    def make_reaction_string(self, nucleophile, electrophile, catalysts, ligands, product):
        reactants = nucleophile + '.' + electrophile + '.' + catalysts + '.' + ligands
        if reactants[-1] == '.':
            reactants = reactants[:-1]
        return reactants + '>>' + product

    def convert_dict_to_yieldbert_labels(self, single_rxn):
        return single_rxn['yield']

    def find_product(self, single_rxn):
        '''
        Each rxn has only one product.
        '''
        product = single_rxn['rx'][0][-1][0]
        product = remove_atom_mapped_numbering(product)
        return product


def main():
    # Init arguments.
    args = init_args()
    paths_to_suzuki_data = args.paths_to_parsed_suzuki_data
    save_path = args.save_path
    split = args.split

    # Load in data.
    suzuki_rxns = load_parsed_data(paths_to_suzuki_data[0], start_index=0)
    suzuki_rxns_2 = load_parsed_data(paths_to_suzuki_data[1], start_index=len(suzuki_rxns))
    suzuki_rxns.update(suzuki_rxns_2)

    del suzuki_rxns[3818], suzuki_rxns[420], suzuki_rxns[2922]

    # Dataframe splits.
    if split in ['catalyst', 'ligand']:
        most_common_catalysts, most_common_ligands = ranking_catalysts_and_ligands(suzuki_rxns)
        if split == 'catalyst':
            train_val_dict, test_dict = split_on_cat_lig(suzuki_rxns, most_common_catalysts[
                0])  # if using multiple electrophile splits it's 0:35, and 35:
            train_dict, val_dict = split_on_cat_lig(train_val_dict, most_common_catalysts[1])
        else:
            train_val_dict, test_dict = split_on_cat_lig(suzuki_rxns, most_common_ligands[
                0])  # if using multiple electrophile splits it's 0:35, and 35:
            train_dict, val_dict = split_on_cat_lig(train_val_dict, most_common_ligands[1])

    elif split in ['nucleophile', 'electrophile']:
        most_common_nucleophiles, most_common_electrophiles = ranking_nucleophiles_and_electrophiles(suzuki_rxns)
        if split == 'nucleophlie':
            train_val_dict, test_dict = split_on_reactant(suzuki_rxns, most_common_nucleophiles[0])
            train_dict, val_dict = split_on_reactant(train_val_dict, most_common_nucleophiles[1])
        else:
            train_val_dict, test_dict = split_on_multiple_electrophiles(suzuki_rxns, most_common_electrophiles[0:35])
            train_dict, val_dict = split_on_multiple_electrophiles(train_val_dict, most_common_electrophiles[35:])

    else:
        assert split in ['nucleophile', 'electrophile', 'catalyst', 'ligand']

    # Creating the train_df.
    train_text = []
    train_labels = []
    for key in list(train_dict.keys()):
        train_text.append(YieldBert_Setup().convert_dict_to_yieldbert_text(train_dict[key]))
        train_labels.append(YieldBert_Setup().convert_dict_to_yieldbert_labels(train_dict[key]))

    train_df = pd.DataFrame({'text' : train_text, 'labels' : train_labels})
    mean = np.mean(train_df['labels'])
    std = np.std(train_df['labels'])
    train_df['labels'] = (train_df['labels'] - mean) / std

    # Creating the val_df.
    val_text = []
    val_labels = []
    for key in list(val_dict.keys()):
        val_text.append(YieldBert_Setup().convert_dict_to_yieldbert_text(val_dict[key]))
        val_labels.append(YieldBert_Setup().convert_dict_to_yieldbert_labels(val_dict[key]))

    val_df = pd.DataFrame({'text': val_text, 'labels': val_labels})
    print(val_text[0])
    print(val_labels[0])
    val_df['labels'] = (val_df['labels'] - mean) / std

    # Creating the test_df.
    test_text = []
    test_labels = []
    for key in list(test_dict.keys()):
        test_text.append(YieldBert_Setup().convert_dict_to_yieldbert_text(test_dict[key]))
        test_labels.append(YieldBert_Setup().convert_dict_to_yieldbert_labels(test_dict[key]))

    test_df = pd.DataFrame({'text': test_text, 'labels': test_labels})
    test_df['labels'] = (test_df['labels'] - mean) / std

    yield_bert.train_model(train_df, output_dir=save_path, eval_df=test_df)

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

    yield_predicted = trained_yield_bert.predict(test_df.text.values)[0] * std + mean

    yield_true = test_df.labels.values * std + mean

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
