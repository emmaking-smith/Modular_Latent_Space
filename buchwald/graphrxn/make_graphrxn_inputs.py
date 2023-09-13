'''
Making these graphrxn inputs
'''

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
sys.path.append('..')
from buchwald_splits import split_on_molecule, split_on_molecules, find_mol_ranking

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def make_stars(df_row):
    halide = df_row['aryl_halide_smiles']
    base = df_row['base_smiles']
    ligand = df_row['ligand_smiles']
    additive = df_row['additive_smiles']
    if halide == '':
        product = 'NC1=CC=C(C)C=C1'
    else:
        product = make_product(halide, 'NC1=CC=C(C)C=C1')
    rxn_smiles = canonicalize_smiles(ligand) + '*' + canonicalize_smiles(halide) + '*' + canonicalize_smiles(base) + '*' + canonicalize_smiles(additive) + '*' + canonicalize_smiles(product)
    return rxn_smiles

def make_product(electrophile, nucleophile):
    electrophile = Chem.MolFromSmiles(electrophile)
    nucleophile = Chem.MolFromSmiles(nucleophile)
    rxn = AllChem.ReactionFromSmarts('[c:1][Cl,Br,I:2].[c:3]-[N:4]>>[c:1][c:3]')
    product = Chem.MolToSmiles(rxn.RunReactants((electrophile, nucleophile))[0][0])
    return product

def make_reaction_column(df):
    reactions = []
    for i in range(len(df)):
        reactions.append(make_stars(df.iloc[i]))
    return reactions

def make_labels(df):
    return np.array(df['yield'].tolist())

def make_graphrxn_input(df, save_path):
    new_df = pd.DataFrame()
    new_df['reaction'] = make_reaction_column(df)
    labels = make_labels(df)
    mean = np.mean(labels)
    std = np.std(labels)
    scaled = [((x - mean) / std) for x in labels]
    new_df['temp'] = np.nan
    new_df['output'] = scaled
    new_df['origin_output'] = labels
    if save_path is not None:
        new_df.to_csv(save_path, index=False)
        return print(f'Saved to {save_path}')
    else:
        return new_df

def remove_nans(df):
    df.loc[pd.isna(df['aryl_halide_smiles']) == True, ['aryl_halide_smiles']] = ''
    df.loc[pd.isna(df['ligand_smiles']) == True, ['ligand_smiles']] = ''
    df.loc[pd.isna(df['base_smiles']) == True, ['base_smiles']] = ''
    df.loc[pd.isna(df['additive_smiles']) == True, ['additive_smiles']] = ''
    return df

def main():
    split = 'halide'
    test_mol_idx = 0
    print('Loading in data...')
    print()

    # Import the datasets.
    buchwald_rxns = pd.read_csv('doyle_buchwald_data.csv')
    buchwald_rxns = remove_nans(buchwald_rxns)

    if split in ['ligand', 'base']:
        dataframe_splits = split_on_molecule

    else:
        dataframe_splits = split_on_molecules

    print('Splitting data...')
    print()
    # Splitting the Dataframe
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

    print('Making GraphRxn Dataframe...')
    print()
    name = 'buchwald_' + split + '_' + str(test_mol_idx) + '_graphrxn_train.csv'
    make_graphrxn_input(train_df, name)
    name = 'buchwald_' + split + '_' + str(test_mol_idx) + '_graphrxn_test.csv'
    make_graphrxn_input(test_df, name)

if __name__ == '__main__':
    main()