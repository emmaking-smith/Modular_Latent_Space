'''
This damn GraphRXN is a pain in butt.
'''

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from suzuki_splits import load_parsed_data, ranking_nucleophiles_and_electrophiles, \
    split_on_reactant, split_on_multiple_electrophiles, split_on_cat_lig, \
    ranking_catalysts_and_ligands, find_reactants, find_catalysts_ligands, reformat_ligs_and_cats, clean_ligands

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def make_stars(rxn_dictionary, index):
    nucleophile, electrophile = find_reactants(rxn_dictionary[index]['rx'])
    catalyst, ligand = find_catalysts_ligands(rxn_dictionary[index])
    catalyst = reformat_ligs_and_cats([catalyst])
    ligand = clean_ligands([ligand])
    ligand = reformat_ligs_and_cats(ligand)
    if index == 6009:
        product = make_product(nucleophile, electrophile)
    else:
        product = make_product(electrophile, nucleophile)

    rxn_smiles = canonicalize_smiles(electrophile) + '*' + canonicalize_smiles(nucleophile) + '*' + canonicalize_smiles(catalyst[0]) + '*' + canonicalize_smiles(ligand[0]) + '*' + canonicalize_smiles(product)
    return rxn_smiles

def make_product(electrophile, nucleophile):
    electrophile = Chem.MolFromSmiles(electrophile)
    nucleophile = Chem.MolFromSmiles(nucleophile)

    rxn = AllChem.ReactionFromSmarts('[c:1][Cl,Br,I:2].[c:3]-[B:4]>>[c:1][c:3]')
    product = Chem.MolToSmiles(rxn.RunReactants((electrophile, nucleophile))[0][0])
    return product

def make_reaction_column(dict):
    reactions = []
    for i in list(dict.keys()):
        reactions.append(make_stars(dict, i))
    return reactions

def make_labels(rxn_dict):
    labels = []
    for i in list(rxn_dict.keys()):
        labels.append(rxn_dict[i]['yield'])
    return np.array(labels)

def make_graphrxn_input(dict, save_path):
    df = pd.DataFrame()
    df['reaction'] = make_reaction_column(dict)
    labels = make_labels(dict)
    mean = np.mean(labels)
    std = np.std(labels)
    scaled = [((x - mean) / std) for x in labels]
    df['temp'] = np.nan
    df['output'] = scaled
    df['origin_output'] = labels
    if save_path is not None:
        df.to_csv(save_path, index=False)
        return print(f'Saved to {save_path}')
    else:
        return df

def main():
    split = 'electrophile'

    print('Loading in data...')
    print()
    # Import the datasets.
    data_paths = ['suzuki_from_arom_USPTO_parsed_het.txt', 'suzuki_from_arom_USPTO_parsed_homo.txt']
    suzuki_rxns = load_parsed_data(data_paths[0], start_index=0)
    suzuki_rxns_2 = load_parsed_data(data_paths[1], start_index=len(suzuki_rxns))
    suzuki_rxns.update(suzuki_rxns_2)

    # Deleting the rxns with super long catalysts.
    del suzuki_rxns[3818], suzuki_rxns[420], suzuki_rxns[2922]

    # Dataframe splits.
    most_common_nucleophiles, most_common_electrophiles = ranking_nucleophiles_and_electrophiles(suzuki_rxns)
    if split == 'nucleophlie':
        train_val_dict, test_dict = split_on_reactant(suzuki_rxns, most_common_nucleophiles[0])
        train_dict, val_dict = split_on_reactant(train_val_dict, most_common_nucleophiles[1])
    else:
        train_val_dict, test_dict = split_on_multiple_electrophiles(suzuki_rxns, most_common_electrophiles[0:35])
        train_dict, val_dict = split_on_multiple_electrophiles(train_val_dict, most_common_electrophiles[35:])

    make_graphrxn_input(train_dict, 'suzuki_electrophile_graphrxn_train.csv')
    make_graphrxn_input(test_dict, 'suzuki_electrophile_graphrxn_test.csv')


if __name__ == '__main__':
    main()
