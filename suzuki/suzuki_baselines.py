'''
Baseline ML models.
'''

import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from sklearn.decomposition import PCA
from suzuki_splits import load_parsed_data, ranking_nucleophiles_and_electrophiles, \
    split_on_reactant, split_on_multiple_electrophiles, split_on_cat_lig, \
    ranking_catalysts_and_ligands, find_reactants, find_catalysts_ligands, reformat_ligs_and_cats, clean_ligands

rf = RandomForestRegressor()
gp = GaussianProcessRegressor(kernel=Matern())
adaboost = AdaBoostRegressor()
pca = PCA(n_components=1000)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, help='Options are: nucleophile, electrophile, catalyst, ligand.')
    return parser.parse_args()

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def make_rxn_smiles(rxn_dictionary, index):
    nucleophile, electrophile = find_reactants(rxn_dictionary[index]['rx'])
    catalyst, ligand = find_catalysts_ligands(rxn_dictionary[index])
    catalyst = reformat_ligs_and_cats([catalyst])
    ligand = clean_ligands([ligand])
    ligand = reformat_ligs_and_cats(ligand)

    rxn_smiles = canonicalize_smiles(nucleophile) + '.' + canonicalize_smiles(electrophile) + '.' + canonicalize_smiles(catalyst[0]) + '.' + canonicalize_smiles(ligand[0])
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

def make_fingerprint(rxn_dict, index):
    rxn_smiles = make_rxn_smiles(rxn_dict, index)
    rxn_smiles = remove_double_dots(remove_terminal_dots(rxn_smiles))
    if rxn_smiles == '.':
        rxn_smiles = ''
    mols = Chem.MolFromSmiles(rxn_smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(mols, 2, 2048)
    return np.array(fps)

def make_inputs(rxn_dict):
    inputs = np.array([])
    for i in list(rxn_dict.keys()):
        rxn_fps_i = make_fingerprint(rxn_dict, i)
        inputs = np.concatenate((inputs, rxn_fps_i))
    return inputs.reshape([len(rxn_dict), -1])

def MAE(true, pred):
    return np.mean(np.abs(true - pred))

def gp_MAE(true, pred, scaler):
    pred = scaler.inverse_transform(pred).reshape([-1])
    mae = MAE(true, pred)
    return mae

def make_labels(rxn_dict):
    labels = []
    for i in list(rxn_dict.keys()):
        labels.append(rxn_dict[i]['yield'])
    return np.array(labels)

def main():
    args = init_args()
    split = args.split
    # split = 'electrophile'

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
    if split in ['catalyst', 'ligand']:
        most_common_catalysts, most_common_ligands = ranking_catalysts_and_ligands(suzuki_rxns)
        if split == 'catalyst':
            train_val_dict, test_dict = split_on_cat_lig(suzuki_rxns, most_common_catalysts[0])  # if using multiple electrophile splits it's 0:35, and 35:
            train_dict, val_dict = split_on_cat_lig(train_val_dict, most_common_catalysts[1])
        else:
            train_val_dict, test_dict = split_on_cat_lig(suzuki_rxns, most_common_ligands[0])  # if using multiple electrophile splits it's 0:35, and 35:
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

    print('Embedding the molecules...')
    print()

    # Make the inputs and labels
    train_inputs = make_inputs(train_dict)
    test_inputs = make_inputs(test_dict)
    all_inputs = pca.fit_transform(np.concatenate((train_inputs, test_inputs)))
    train_inputs = all_inputs[0:len(train_dict), :]
    test_inputs = all_inputs[len(train_dict):, :]

    train_labels = make_labels(train_dict)
    test_labels = make_labels(test_dict)

    print(f"------------- PREDICTIONS FOR {split} -------------")
    print()
    rf.fit(train_inputs, train_labels)
    rf_pred = rf.predict(test_inputs)

    loss = np.abs(test_labels - rf_pred)
    print (f"RF Test MAE: {np.mean(loss)}")
    # print(f"RF Median: {np.median(loss)}")
    # print(f"RF 'Chemist' MAE: {np.mean([0 if x <= 10 else x for x in loss])}")
    # print()

    adaboost.fit(train_inputs, train_labels)
    adaboost_pred = adaboost.predict(test_inputs)

    loss = np.abs(test_labels - adaboost_pred)
    print(f"Adaboost Test MAE: {np.mean(loss)}")
    # print(f"Adaboost Median: {np.median(loss)}")
    # print(f"Adaboost 'Chemist' MAE: {np.mean([0 if x <= 10 else x for x in loss])}")
    # print()

    # gp.fit(train_inputs, train_labels)
    # gp_pred = gp.predict(test_inputs)
    # loss = np.abs(test_labels - gp_pred)
    # print(f"GP Test MAE: {np.mean(loss)}")
    # print(f"GP Median: {np.median(loss)}")
    # print(f"GP 'Chemist' MAE: {np.mean([0 if x <= 10 else x for x in loss])}")
    # print()

if __name__ == '__main__':
    main()