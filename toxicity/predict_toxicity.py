'''
Molecule toxicity prediction with Big MPNN. LD50.
'''

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import logging
import argparse
import os
from pathlib import Path
from mpnn_toxicity import Toxicity_MPNN
from pytorch_toxicity import Toxic_NonToxic_Molecules, collate_fn, canonicalize_smiles
from rdkit import Chem

class Initialization:
    def __init__(self):
        pass

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_to_toxicity_data', nargs='+',
                            help='The path to the TDC LD50 toxicity train, val, and test pickle file.',
                            default=['data/TDC_toxicity_train.pickle',
                                     'data/TDC_toxicity_val.pickle',
                                     'data/TDC_toxicity_test.pickle'])
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--message_size', type=int, default=128)
        parser.add_argument('--message_passes', type=int, default=3)
        parser.add_argument('--pretrained_mpnn_path', type=str, default='../big_mpnn_no_delocalised_no_unknown_model')
        parser.add_argument('--save_path', '-s', type=str)
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

def canonicalize_df(df):
    df['canonical_smiles'] = [canonicalize_smiles(x) for x in df['Drug']]
    return df

def find_mol_sizes(df):
    df['mol_size'] = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in df['canonical_smiles']]
    return df

def main():
    args = Initialization().init_args()
    path_to_toxicity_data = args.path_to_toxicity_data
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    message_size = args.message_size
    message_passes = args.message_passes
    pretrained_mpnn_path = args.pretrained_mpnn_path
    save_path = args.save_path

    atom_list = [6, 8, 7, 9, 17, 16, 15, 5, 29, 35, 14, 53, 30, 26, 27, 28, 48, 44, 42, 25, 47, 34, 46, 78, 50, 74,
                 11, 3, 19, 23, 13, 79, 45, 82, 75, 77, 76, 51, 24, 33, 22, 32, 80, 92, 31, 63, 52, 12, 40, 65, 83,
                 49, 64, 20, 66, 57, 60, 62, 39, 56, 68, 59, 58, 70, 55, 38, 41, 37, 73, 67, 21, 81, 71, 72, 90, 69,
                 43, 4, 93, 94, 54, 95, 2, 10, 18, 98, 36, 96, 97, 91]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = Initialization().logging_setup()
    logger.debug('Our device is %s', device)

    # Loading in data
    train = pd.read_pickle(path_to_toxicity_data[0])
    val = pd.read_pickle(path_to_toxicity_data[1])
    test = pd.read_pickle(path_to_toxicity_data[2])

    train = canonicalize_df(train)
    val = canonicalize_df(val)
    test = canonicalize_df(test)

    train = find_mol_sizes(train)
    val = find_mol_sizes(val)
    test = find_mol_sizes(test)
    preds = test.copy()

    # Find longest molecule
    longest_molecule = max(max(train['mol_size']), max(val['mol_size']), max(test['mol_size'])) + 1

    logger.debug('Longest Molecule: %d', longest_molecule)
    logger.debug('Training on %d molecules, validating on %d molecules, testing on %d molecules', len(train),
                 len(val), len(test))

    # Setting up the Pytorch datasets and dataloader.
    train_dataset = Toxic_NonToxic_Molecules(train, longest_molecule)
    val_dataset = Toxic_NonToxic_Molecules(val, longest_molecule)
    test_dataset = Toxic_NonToxic_Molecules(test, longest_molecule)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Defining the model and optimizer.
    model = Toxicity_MPNN(message_size=message_size, message_passes=message_passes, atom_list=atom_list,
                          pretrained_mpnn_path=pretrained_mpnn_path, longest_molecule=longest_molecule)

    logger.debug('Model defined. Message Size: %d, Message Passes: %d, Atom List: %s, Pretrained Model Path: %s',
                 message_size, message_passes, atom_list, pretrained_mpnn_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    del train_dataset, val_dataset, test_dataset, atom_list

    model.to(device).float()

    for epoch in range(epochs):
        batch_train_loss = []
        batch_val_loss = []
        model.train()
        for i, (mol_features, mol_matrices, tox_labels) in enumerate(train_dataloader):
            mol_features = mol_features.to(device).float()
            mol_matrices = mol_matrices.to(device).float()
            tox_labels = tox_labels.to(device).float()

            optimizer.zero_grad()

            pred_toxicities = model(mol_matrices, mol_features)

            loss = torch.nn.L1Loss()(pred_toxicities, tox_labels)
            batch_train_loss.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

        # Finding the average train loss.
        logger.debug('Mean Train Loss  at epoch %d is %.4f', epoch, np.mean(batch_train_loss))

        model.eval()
        with torch.no_grad():
            for i, (mol_features, mol_matrices, tox_labels) in enumerate(val_dataloader):
                mol_features = mol_features.to(device).float()
                mol_matrices = mol_matrices.to(device).float()
                tox_labels = tox_labels.to(device).float()

                pred_toxicities = model(mol_matrices, mol_features)

                loss = torch.nn.L1Loss()(pred_toxicities, tox_labels)
                batch_val_loss.append(loss.cpu().detach().numpy())
        # Finding the average train loss.
        logger.debug('Mean Val Loss epoch %d is %.4f', epoch,np.mean(batch_val_loss))

    # Evaluating on test set.
    logger.debug('Training Finished. Starting evaluation on test set.')

    model.eval()
    predicted_tox = []
    with torch.no_grad():
        for i, (mol_features, mol_matrices, tox_labels) in enumerate(test_dataloader):
            mol_features = mol_features.to(device).float()
            mol_matrices = mol_matrices.to(device).float()

            pred_toxicities = model(mol_matrices, mol_features)
            predicted_tox = predicted_tox + list(pred_toxicities.cpu().detach().numpy())

    preds['predicted_toxicity'] = predicted_tox

    pred_tensor = torch.tensor(preds['predicted_toxicity'].tolist()).float()
    true_tensor = torch.tensor(preds['Y'].tolist()).float()
    loss = torch.nn.L1Loss()(pred_tensor, true_tensor)

    logger.debug('Final Overall Test Loss is %.4f', loss.cpu().detach().numpy())

    # Saving out the models and predictions.
    pred_save_path = os.path.join(save_path, 'preds.pickle')
    model_save_path = os.path.join(save_path, 'model')

    torch.save(model.state_dict(), model_save_path)
    preds.to_pickle(pred_save_path)

if __name__ == '__main__':
    main()




