'''
Predicting the fragrance of enantiomers from AI Crowd dataset.

Pt 2 - Add in the the enantiomers, leave out the idxs selected in chemdraw.
'''

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import os
from pathlib import Path
from mpnn_fragrance import Fragrance_MPNN
from pytorch_fragrance import Odor_Classification, collate_fn
from sklearn.metrics import classification_report, accuracy_score

class Initialization:
    def __init__(self):
        pass

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_to_fragrance_data', type=str,
                            help='The path to the Pyrfume pickle pickle file.',
                            default='data/pyrfume_enantiomer_pairs_removed.pickle')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--message_size', type=int, default=128)
        parser.add_argument('--message_passes', type=int, default=3)
        parser.add_argument('--pretrained_mpnn_path', type=str, default='../MPNN/big_mpnn_no_delocalised_no_unknown_model')
        parser.add_argument('--save_path', '-s', type=str)
        parser.add_argument('--test_data_path', type=str, default='data/one_hot_aicrowd_enantiomers_columns_fixed.pickle',)
        return parser.parse_args()

    def logging_setup(self):
        args = self.init_args()
        save_path = args.save_path
        # Make the save_path
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Set up the logger.
        log_path = os.path.join(save_path, 'model_log.log')
        logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
        logger = logging.getLogger()

        # Setting the threshold of logger to DEBUG
        logger.setLevel(logging.DEBUG)
        return logger

# Functions that predict the top-N labels & yield the true labels.
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

def simplify(pred):
    simplified = np.zeros(len(pred))
    for i, entry in enumerate(pred):
        if entry > 0.5:
            simplified[i] = 1
    return simplified

def format_all_preds(list_of_preds):
    all_preds = np.array([])
    for pred in list_of_preds:
        all_preds = np.concatenate((all_preds, simplify(pred)))
    return all_preds.reshape([-1])

def scoring(true, pred):
    pred = format_all_preds(pred)
    true = np.array(true).reshape([-1])
    f1 = classification_report(true, pred, output_dict=True, zero_division=0.0)
    f1_macro = f1['macro avg']['f1-score']
    f1_weighted = f1['weighted avg']['f1-score']
    accuracy = accuracy_score(true, pred)
    return f1_macro, f1_weighted, accuracy

def main():
    args = Initialization().init_args()
    path_to_fragrance_data = args.path_to_fragrance_data
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    message_size = args.message_size
    message_passes = args.message_passes
    pretrained_mpnn_path = args.pretrained_mpnn_path
    save_path = args.save_path
    test_data_path = args.test_data_path

    atom_list = [6, 8, 7, 9, 17, 16, 15, 5, 29, 35, 14, 53, 30, 26, 27, 28, 48, 44, 42, 25, 47, 34, 46, 78, 50, 74,
                 11, 3, 19, 23, 13, 79, 45, 82, 75, 77, 76, 51, 24, 33, 22, 32, 80, 92, 31, 63, 52, 12, 40, 65, 83,
                 49, 64, 20, 66, 57, 60, 62, 39, 56, 68, 59, 58, 70, 55, 38, 41, 37, 73, 67, 21, 81, 71, 72, 90, 69,
                 43, 4, 93, 94, 54, 95, 2, 10, 18, 98, 36, 96, 97, 91]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = Initialization().logging_setup()
    logger.debug('Our device is %s', device)

    # Loading in data
    df = pd.read_pickle(path_to_fragrance_data)
    enantiomers = pd.read_pickle(test_data_path)

    # Find longest molecule
    longest_molecule = max([Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in df['canonical_smiles']]) + 1

    test_idxs = [6, 61, 8, 86, 10, 51, 20, 38, 48, 16, 9, 34, 12, 74, 13, 27, 17, 85, 39, 42, 46, 68]

    # Split the dataframe.
    test_df = enantiomers.loc[test_idxs].reset_index(drop=True)
    enantiomers = enantiomers.drop(test_idxs)
    train_df = pd.concat((df, enantiomers)).reset_index(drop=True)
    # train_df = enantiomers.reset_index(drop=True)
    preds = test_df.copy()

    logger.debug('Longest Molecule: %d', longest_molecule)
    logger.debug('Training on %d molecules, testing on %d enantiomers', len(train_df),
                 len(test_df))

    # Setting up the Pytorch datasets and dataloader.
    train_dataset = Odor_Classification(train_df, longest_molecule)
    test_dataset = Odor_Classification(test_df, longest_molecule)

    _, fragrance_classes_0 = train_dataset[0]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Defining the model and optimizer.
    model = Fragrance_MPNN(message_size=message_size, message_passes=message_passes, atom_list=atom_list,
                          pretrained_mpnn_path=pretrained_mpnn_path,
                           number_of_labels=fragrance_classes_0.size()[0])

    logger.debug('Model defined. Message Size: %d, Message Passes: %d, Atom List: %s, Pretrained Model Path: %s, Number of Labels: %d',
                 message_size, message_passes, atom_list, pretrained_mpnn_path, fragrance_classes_0.size()[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    del train_dataset, test_dataset, atom_list

    model.to(device).float()

    for epoch in range(epochs):
        batch_train_loss = []
        model.train()
        for i, (mol_features, mol_matrices, fragrance_labels) in enumerate(train_dataloader):
            mol_features = mol_features.to(device).float()
            mol_matrices = mol_matrices.to(device).float()
            fragrance_labels = fragrance_labels.to(device).float()

            optimizer.zero_grad()

            pred_odor = model(mol_matrices, mol_features)

            loss = torch.nn.CrossEntropyLoss()(pred_odor, fragrance_labels)
            batch_train_loss.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

        # Finding the average train loss.
        logger.debug('Mean Train Loss  at epoch %d is %.4f', epoch, np.mean(batch_train_loss))

    # Evaluating on test set.
    logger.debug('Training Finished. Starting evaluation on test set.')

    model.eval()
    predicted_odor_classes = []
    true_odor_classes = []
    with torch.no_grad():
        for i, (mol_features, mol_matrices, fragrance_labels) in enumerate(test_dataloader):
            mol_features = mol_features.to(device).float()
            mol_matrices = mol_matrices.to(device).float()

            pred_odor = model(mol_matrices, mol_features)
            predicted_odor_classes = predicted_odor_classes + list(pred_odor.cpu().detach().numpy())
            true_odor_classes = true_odor_classes + list(fragrance_labels.cpu().detach().numpy())

    pred_tensor = torch.tensor(predicted_odor_classes).float()
    true_tensor = torch.tensor(true_odor_classes).float()
    loss = torch.nn.CrossEntropyLoss()(pred_tensor, true_tensor)

    logger.debug('Final Overall Test Loss is %.4f', loss.cpu().detach().numpy())

    # Finding the Multiclass F1 & Balanced Accuracy Scores.
    f1_macro, f1_weighted, accuracy = scoring(true_odor_classes, predicted_odor_classes)
    logger.debug('F1 (Macro): %.2f, F1 (Weighted): %.2f, Accuracy: %.2f', f1_macro, f1_weighted, accuracy)

    # Generating the true / pred labels.
    true_labels = make_true_labels(preds)
    pred_labels = make_top_N_pred_labels(preds, np.array(predicted_odor_classes))

    preds['true_labels'] = true_labels
    preds['pred_labels'] = pred_labels

    # Saving out the predictions.
    pred_save_path = os.path.join(save_path, 'preds.pickle')
    model_save_path = os.path.join(save_path, 'model')

    torch.save(model.state_dict(), model_save_path)
    preds.to_pickle(pred_save_path)

if __name__ == '__main__':
    main()
