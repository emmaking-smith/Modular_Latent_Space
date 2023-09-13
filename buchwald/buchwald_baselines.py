'''
Baseline ML models.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from sklearn.decomposition import PCA
from buchwald_splits import split_on_molecule, split_on_molecules, find_mol_ranking

rf = RandomForestRegressor()
gp = GaussianProcessRegressor(kernel=Matern())
adaboost = AdaBoostRegressor()
pca = PCA(n_components=1000)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, help='Options are: halide, ligand, base, additive.')
    parser.add_argument('--test_mol_idx', type=int, help='What molecule to leave out for testing.')
    return parser.parse_args()

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def make_rxn_smiles(df_row):
    halide = df_row['aryl_halide_smiles']
    base = df_row['base_smiles']
    ligand = df_row['ligand_smiles']
    additive = df_row['additive_smiles']
    rxn_smiles = canonicalize_smiles(halide) + '.' + canonicalize_smiles(base) + '.' + canonicalize_smiles(ligand) + '.' + canonicalize_smiles(additive)
    return rxn_smiles

def remove_double_dots(rxn_smiles):
    rxn_smiles = rxn_smiles.replace('..', '.')
    return rxn_smiles

def remove_terminal_dots(rxn_smiles):
    if '.' == rxn_smiles[0]:
        rxn_smiles = rxn_smiles[1:]
    if '.' == rxn_smiles[-1]:
        rxn_smiles = rxn_smiles[0:-1]
    return rxn_smiles

def make_fingerprint(df_row):
    rxn_smiles = make_rxn_smiles(df_row)
    rxn_smiles = remove_double_dots(remove_terminal_dots(rxn_smiles))
    if rxn_smiles == '.':
        rxn_smiles = ''
    mols = Chem.MolFromSmiles(rxn_smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(mols, 2, 2048)
    return np.array(fps)

def make_inputs(df):
    inputs = np.array([])
    for i in range(len(df)):
        rxn_fps_i = make_fingerprint(df.iloc[i])
        inputs = np.concatenate((inputs, rxn_fps_i))
    return inputs.reshape([len(df), -1])

def MAE(true, pred):
    return np.mean(np.abs(true - pred))

def gp_MAE(true, pred, scaler):
    pred = scaler.inverse_transform(pred).reshape([-1])
    mae = MAE(true, pred)
    return mae

def remove_nans(df):
    df.loc[pd.isna(df['aryl_halide_smiles']) == True, ['aryl_halide_smiles']] = ''
    df.loc[pd.isna(df['ligand_smiles']) == True, ['ligand_smiles']] = ''
    df.loc[pd.isna(df['base_smiles']) == True, ['base_smiles']] = ''
    df.loc[pd.isna(df['additive_smiles']) == True, ['additive_smiles']] = ''
    return df

def main():
    # Initialize the arguments
    args = init_args()
    split = args.split
    test_mol_idx = args.test_mol_idx

    # split = 'additive'
    # test_mol_idx = 3

    print('Loading in data...')
    print()
    # Import the datasets.
    buchwald_rxns = pd.read_csv('doyle_buchwald_data.csv')
    buchwald_rxns = remove_nans(buchwald_rxns)

    if split in ['ligand', 'base']:
        dataframe_splits = split_on_molecule

    else:
        dataframe_splits = split_on_molecules

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

    print('Embedding the molecules...')
    print()

    # Make the inputs and labels

    train_inputs =pca.fit_transform(make_inputs(train_df))
    test_inputs = pca.fit_transform(make_inputs(test_df))

    print(f"------------- PREDICTIONS FOR {split} {test_mol_idx} -------------")
    print()
    rf.fit(train_inputs, train_df['yield'])
    rf_pred = rf.predict(test_inputs)

    loss = np.abs(test_df['yield'] - rf_pred)
    print (f"RF Test MAE: {np.mean(loss)}")
    # print(f"RF Median: {np.median(loss)}")
    # print(f"RF 'Chemist' MAE: {np.mean([0 if x <= 10 else x for x in loss])}")
    print()

    adaboost.fit(train_inputs, train_df['yield'])
    adaboost_pred = adaboost.predict(test_inputs)

    loss = np.abs(test_df['yield'] - adaboost_pred)
    print(f"Adaboost Test MAE: {np.mean(loss)}")
    # print(f"Adaboost Median: {np.median(loss)}")
    # print(f"Adaboost 'Chemist' MAE: {np.mean([0 if x <= 10 else x for x in loss])}")
    print()

    gp.fit(train_inputs, train_df['yield'])
    gp_pred = gp.predict(test_inputs)

    loss = np.abs(test_df['yield'] - gp_pred)
    print(f"GP Test MAE: {np.mean(loss)}")
    # print(f"GP Median: {np.median(loss)}")
    # print(f"GP 'Chemist' MAE: {np.mean([0 if x <= 10 else x for x in loss])}")
    print()

if __name__ == '__main__':
    main()