'''
Baseline on TDC toxicity.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem

rf = RandomForestRegressor()
gp = GaussianProcessRegressor(kernel=Matern())
adaboost = AdaBoostRegressor()
pca = PCA(n_components=1000)

def make_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return np.array(fps)

def make_inputs(df):
    inputs = np.array([])
    for smiles in df['canonical_smiles']:
        fps_i = make_fingerprint(smiles)
        inputs = np.concatenate((inputs, fps_i))
    return inputs.reshape([len(df), -1])

def canonicalize_df(df):
    df['canonical_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in df['Drug']]
    return df

def MAE(true, pred):
    return np.mean(np.abs(true - pred))

def gp_MAE(true, pred, scaler):
    pred = scaler.inverse_transform(pred).reshape([-1])
    mae = MAE(true, pred)
    return mae

def main():

    print('Loading in data...')
    print()
    # Import the datasets.
    train = pd.read_pickle('data/TDC_toxicity_train.pickle')
    test = pd.read_pickle('data/TDC_toxicity_test.pickle')
    # test = pd.read_pickle('data/new_compounds_for_testing_TDC.pickle')

    train = canonicalize_df(train)
    test = canonicalize_df(test)

    print('Embedding the molecules...')
    print()

    # Make the inputs and labels
    train_inputs = make_inputs(train)
    test_inputs = make_inputs(test)
    # new_test_inputs = make_inputs(new_test)

    all_inputs = np.concatenate((train_inputs, test_inputs))
    all_inputs = pca.fit_transform(all_inputs)
    train_inputs = all_inputs[0:len(train), :]
    test_inputs = all_inputs[len(train):, :]

    rf.fit(train_inputs, train['Y'])
    rf_pred = rf.predict(test_inputs)
    print (f"RF Test MAE: {MAE(test['Y'], rf_pred)}")

    adaboost.fit(train_inputs, train['Y'])
    adaboost_pred = adaboost.predict(test_inputs)
    print(f"Adaboost Test MAE: {MAE(test['Y'], adaboost_pred)}")

    scaler = StandardScaler().fit(np.array(test['Y']).reshape(-1,1))

    gp.fit(StandardScaler().fit_transform(train_inputs), StandardScaler().fit_transform(np.array(train['Y']).reshape(-1,1)))
    gp_pred = gp.predict(StandardScaler().fit_transform(test_inputs))
    print(f"GP Test MAE: {gp_MAE(test['Y'], gp_pred, scaler)}")

if __name__ == '__main__':
    main()