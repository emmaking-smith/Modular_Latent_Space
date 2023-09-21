'''
Predicting on the ~10 enantiomer pairs selected in chemdraw w/ JUST RF and KNN.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
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
        chiral_tag_i = make_chiral_tag(smiles)
        fps_i = np.concatenate((fps_i, chiral_tag_i))
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
    parser.add_argument('--path_to_enantiomers', type=str, help='The path to the enantiomer pickle file.',
                        default='data/one_hot_aicrowd_enantiomers_columns_fixed.pickle')
    return parser.parse_args()

def make_true_label(vocab, df_row):
    '''
    From a single row, make the corresponding text label.
    '''
    label = []
    # For every '1' in the df_row, add it to the label
    for name in vocab:
        if df_row[name] == 1:
            label.append(name)
    return label

def make_true_labels(df):
    all_labels = []
    vocab = list(df.keys())[5:-1]
    for i in range(len(df)):
        all_labels.append(make_true_label(vocab, df.iloc[i]))
    return all_labels

def make_top_N_single_pred_label(vocab, pred, true):
    '''
    Find the top N a single predicted label, where N = number of classes in true label.
    '''
    pred_label = []
    N = 5
    # Find top N, including duplicates, then remove the duplicates for the for loop, then sort again to keep order.
    top_Ns = np.sort(np.unique(np.sort(pred)[::-1][0:N]))[::-1]

    # Finding the top N entries.
    top_N_idxs = []
    for top_pred in top_Ns:
        top_pred_idx = np.where(pred == top_pred)[0]
        # Appending the idxs to
        for idx in top_pred_idx:
            top_N_idxs.append(idx)

    # Finding the corresponding vocab.
    for idx in top_N_idxs:
        pred_label.append(vocab[idx])
    return pred_label

def make_top_N_pred_labels(test_df, all_preds):
    vocab = list(test_df.keys())[5:-1]
    all_pred_labels = []
    for i, pred in enumerate(all_preds):
        true_label = make_true_label(vocab, test_df.iloc[i])
        all_pred_labels.append(make_top_N_single_pred_label(vocab, pred, true_label))
    return all_pred_labels

def make_chiral_tag(smiles):
    mol = Chem.MolFromSmiles(smiles)
    chiral_atoms = [str(x.GetChiralTag()) for x in mol.GetAtoms()]
    if 'CHI_TETRAHEDRAL_CW' in chiral_atoms:
        chiral_tag = np.array([1, 0, 0])
    elif 'CHI_TETRAHEDRAL_CCW' in chiral_atoms:
        chiral_tag = np.array([0, 1, 0])
    else:
        chiral_tag = np.array([0, 0, 1])
    return chiral_tag

def main():
    # args = init_args()
    # path_to_data = args.path_to_data
    # path_to_enantiomers = args.path_to_enantiomers
    path_to_data = 'data/pyrfume_enatiomer_pairs_removed.pickle'
    path_to_enantiomers = 'data/one_hot_aicrowd_enantiomers_columns_fixed.pickle'

    print('Loading in data...')
    print()
    df = pd.read_pickle(path_to_data)
    enantiomers = pd.read_pickle(path_to_enantiomers)

    test_idxs = [6, 61, 8, 86, 10, 51, 20, 38, 48, 16, 9, 34, 12, 74, 13, 27, 17, 85, 39, 42, 46, 68]

    # Split the dataframe.
    test_df = enantiomers.loc[test_idxs].reset_index(drop=True)
    enantiomers = enantiomers.drop(test_idxs)
    train_df = pd.concat((df, enantiomers)).reset_index(drop=True)
    preds = test_df.copy()

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

    rf.fit(train_inputs, train_labels)
    rf_preds = rf.predict(test_inputs)
    f1 = classification_report(test_labels, rf_preds, output_dict=True, zero_division=0.0)
    f1_macro = f1['macro avg']['f1-score']
    f1_weighted = f1['weighted avg']['f1-score']
    accuracy = accuracy_score(test_labels, rf_preds)

    # Making the labels.
    true_labels = make_true_labels(preds)
    rf_pred_labels = make_top_N_pred_labels(preds, rf_preds)

    preds['true_labels'] = true_labels
    preds['rf_pred_labels'] = rf_pred_labels

    print('---------------- Random Forest ----------------')
    print(f"F1 (Macro): {f1_macro}, F1 (Weighted): {f1_weighted}, Accuracy: {accuracy}")
    print()

    kn.fit(train_inputs, train_labels)
    kn_preds = kn.predict(test_inputs)
    f1 = classification_report(test_labels, kn_preds, output_dict=True, zero_division=0.0)
    f1_macro = f1['macro avg']['f1-score']
    f1_weighted = f1['weighted avg']['f1-score']
    accuracy = accuracy_score(test_labels, kn_preds)

    kn_pred_labels = make_top_N_pred_labels(preds, kn_preds)
    preds['knn_pred_labels'] = kn_pred_labels

    print('---------------- K-Neighbors ----------------')
    print(f"F1 (Macro): {f1_macro}, F1 (Weighted): {f1_weighted}, Accuracy: {accuracy}")
    print()

    print(preds['rf_pred_labels'])
    print(preds['knn_pred_labels'])
    # preds.to_pickle('enantiomer_baseline_predictions.pickle')
if __name__ == '__main__':
    main()