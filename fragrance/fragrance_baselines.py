'''
Baseline models on multiclass labeling
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import chain
import argparse

rf = RandomForestClassifier()
kn = KNeighborsClassifier()

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

def extract_fragrance_class(df_row):
    keys = list(df_row.keys())
    one_hot_fragrance = df_row[keys[5:-1]]
    return np.array(one_hot_fragrance)

def make_labels(df):
    labels = np.array([])
    for i in range(len(df)):
        label_i = extract_fragrance_class(df.iloc[i])
        labels = np.concatenate((labels, label_i))
    labels = labels.reshape([len(df), -1])
    labels = labels.astype('int')
    return labels

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', type=str, help='path to fragrance pickle data.',
                        default='data/pyrfume_enatiomer_pairs_removed.pickle')
    parser.add_argument('--fold', type=int, help='The fold number (out of 5 folds) to test on. Values range from 0 to 4.')
    return parser.parse_args()

def main():
    args = init_args()
    path_to_data = args.path_to_data
    fold_number = args.fold

    # path_to_data = 'data/pyrfume_enatiomer_pairs_removed.pickle'
    # fold_number = 4
    # assert fold_number in range(5)

    print('Loading in data...')
    print()
    df = pd.read_pickle(path_to_data)

    # Splitting the dataset into 5 equal-ish parts.
    idxs = list(df.index)
    np.random.seed(12)
    np.random.shuffle(idxs)
    kfolds = np.array_split(idxs, 5)

    # Select the testing fold.
    test_idxs = kfolds[fold_number]
    train_idxs = kfolds[:fold_number] + kfolds[(fold_number + 1):]
    train_idxs = np.array(list(chain.from_iterable(train_idxs)))

    # Split the dataframe.
    train_df = df.loc[train_idxs].reset_index(drop=True)
    test_df = df.loc[test_idxs].reset_index(drop=True)

    print('Embedding the molecules...')
    print()
    # Make the inputs and labels
    train_inputs = make_inputs(train_df)
    test_inputs = make_inputs(test_df)

    print('Inputs Made.')
    print()

    print('Forming labels...')
    print()
    train_labels = make_labels(train_df)
    test_labels = make_labels(test_df)

    print('Labels Made.')
    print()

    # Same Kfolds as NN.
    rf.fit(train_inputs, train_labels)
    preds = rf.predict(test_inputs)
    f1 = classification_report(test_labels, preds, output_dict=True, zero_division=0.0)
    f1_macro = f1['macro avg']['f1-score']
    f1_weighted = f1['weighted avg']['f1-score']
    accuracy = accuracy_score(test_labels, preds)
    print(f'------------- FOLD {fold_number + 1} of 5 -------------')
    print('Random Forest')
    print(f"F1 (Macro): {f1_macro}, F1 (Weighted): {f1_weighted}, Accuracy: {accuracy}")
    print()

    kn.fit(train_inputs, train_labels)
    preds = kn.predict(test_inputs)
    f1 = classification_report(test_labels, preds, output_dict=True, zero_division=0.0)
    f1_macro = f1['macro avg']['f1-score']
    f1_weighted = f1['weighted avg']['f1-score']
    accuracy = accuracy_score(test_labels, preds)

    print('K-Neighbors')
    print(f"F1 (Macro): {f1_macro}, F1 (Weighted): {f1_weighted}, Accuracy: {accuracy}")
    print()

if __name__ == '__main__':
    main()