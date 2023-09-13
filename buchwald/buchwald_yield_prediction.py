'''
Buchwald yield prediction with Big MPNN. Leaving out multiple halides and additives.
'''

import numpy as np
import pandas as pd

from buchwald_splits import find_halide, find_ligand, find_base, find_additive, \
    split_on_molecule, split_on_molecules, remove_nans, find_mol_ranking

import torch
from torch.utils.data import DataLoader
import logging
import argparse
import os
from rdkit import Chem
from pathlib import Path
from buchwald_pytorch_dataset import Buchwald_Rxns, collate_fn
from buchwald_yield_mpnn import Buchwald_MPNN

class Initialization:
    def __init__(self):
        pass

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_to_buchwald_data', type=str, help='The path to the Buchwald csv file.',
                            default='doyle_buchwald_data.csv')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--message_size', type=int, default=128)
        parser.add_argument('--message_passes', type=int, default=3)
        parser.add_argument('--pretrained_mpnn_path', type=str, default='../MPNN/big_mpnn_no_delocalised_no_unknown_model')
        parser.add_argument('--save_path', '-s', type=str)
        parser.add_argument('--split', type=str, help='Options are: halide, ligand, base, additive.')
        parser.add_argument('--test_mol_idx', type=int, help='What molecule to leave out for testing.')
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

class Find_Longest_Molecules():
    def __init__(self, df):
        self.df = df

    def find_longest_halide(self):
        halides = [find_halide(self.df.iloc[x]) for x in list(self.df.index)]
        halides_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in halides]
        return max(halides_lengths)

    def find_longest_ligand(self):
        ligands = [find_ligand(self.df.iloc[x]) for x in list(self.df.index)]
        ligands_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in ligands]
        return max(ligands_lengths)

    def find_longest_base(self):
        bases = [find_base(self.df.iloc[x]) for x in list(self.df.index)]
        bases_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in bases]
        return max(bases_lengths)

    def find_longest_additive(self):
        additives = [find_additive(self.df.iloc[x]) for x in list(self.df.index)]
        additives_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in additives]
        return max(additives_lengths)

def main():
    args = Initialization().init_args()
    path_to_buchwald_data = args.path_to_buchwald_data
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    message_size = args.message_size
    message_passes = args.message_passes
    pretrained_mpnn_path = args.pretrained_mpnn_path
    save_path = args.save_path
    split = args.split
    test_mol_idx = args.test_mol_idx

    assert split in ['halide', 'ligand', 'base', 'additive']

    if split in ['ligand', 'base']:
        dataframe_splits = split_on_molecule

    else:
        dataframe_splits = split_on_molecules

    atom_list = [6, 8, 7, 9, 17, 16, 15, 5, 29, 35, 14, 53, 30, 26, 27, 28, 48, 44, 42, 25, 47, 34, 46, 78, 50, 74,
                 11, 3, 19, 23, 13, 79, 45, 82, 75, 77, 76, 51, 24, 33, 22, 32, 80, 92, 31, 63, 52, 12, 40, 65, 83,
                 49, 64, 20, 66, 57, 60, 62, 39, 56, 68, 59, 58, 70, 55, 38, 41, 37, 73, 67, 21, 81, 71, 72, 90, 69,
                 43, 4, 93, 94, 54, 95, 2, 10, 18, 98, 36, 96, 97, 91]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = Initialization().logging_setup()
    logger.debug('Our device is %s', device)

    # Loading in dataset.
    buchwald_rxns = pd.read_csv(path_to_buchwald_data)
    buchwald_rxns = remove_nans(buchwald_rxns)
    # buchwald_rxns = canonicalize_df(buchwald_rxns)

    longest_halide = Find_Longest_Molecules(buchwald_rxns).find_longest_halide() + 1 # 12
    longest_ligand = Find_Longest_Molecules(buchwald_rxns).find_longest_ligand() + 1 # 47
    longest_base = Find_Longest_Molecules(buchwald_rxns).find_longest_base() + 1 # 22
    longest_additive = Find_Longest_Molecules(buchwald_rxns).find_longest_additive() + 1 # 21

    logger.debug('Data Loaded.')

    # Splitting the Dataframe
    ranked_halides, ranked_ligands, ranked_bases, ranked_additives = find_mol_ranking(buchwald_rxns)
    if split == 'halide':
        test_smiles = ranked_halides[test_mol_idx * 4 : (test_mol_idx + 1) * 4]
    elif split == 'ligand':
        test_smiles = ranked_ligands[test_mol_idx]
    elif split == 'base':
        test_smiles = ranked_bases[test_mol_idx]
    else:
        test_smiles = ranked_additives[test_mol_idx * 6 : (test_mol_idx + 1) * 6]

    train_df, test_df = dataframe_splits(buchwald_rxns, test_smiles, split)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    preds = test_df.copy()

    logger.debug('Leaving out and testing on %s %d (Mols: %s)', split, test_mol_idx, test_smiles)
    logger.debug('Training on %d rxns. Testing on %d rxns.', len(train_df), len(test_df))

    train_dataset = Buchwald_Rxns(train_df, longest_halide, longest_ligand, longest_base, longest_additive)
    test_dataset = Buchwald_Rxns(test_df, longest_halide, longest_ligand, longest_base, longest_additive)

    # Define optimizer and model.
    model = Buchwald_MPNN(message_size, message_passes, atom_list, pretrained_mpnn_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.debug('Model defined. Message Size: %d, Message Passes: %d, Atom List: %s, Pretrained MPNN Path: %s',
                 message_size, message_passes, atom_list, pretrained_mpnn_path)

    # Define the dataloaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    del train_dataset, test_dataset, atom_list

    # Train Loop
    model.to(device).float()

    for epoch in range(epochs):
        batch_train_loss = []
        model.train()
        for i, (halide_features, halide_adj_matrices, ligand_features, ligand_adj_matrices,
                base_features, base_adj_matrices, additive_features,
                additive_adj_matrices, yields) in enumerate(train_dataloader):

            halide_features = halide_features.to(device)
            halide_adj_matrices = halide_adj_matrices.to(device)
            ligand_features = ligand_features.to(device)
            ligand_adj_matrices = ligand_adj_matrices.to(device)
            base_features = base_features.to(device)
            base_adj_matrices = base_adj_matrices.to(device)
            additive_features = additive_features.to(device)
            additive_adj_matrices = additive_adj_matrices.to(device)
            yields = yields.to(device)

            optimizer.zero_grad()

            predicted_yields = model(halide_adj_matrices, halide_features, ligand_adj_matrices,
                                     ligand_features, base_adj_matrices, base_features, additive_adj_matrices,
                                     additive_features)

            loss = torch.nn.L1Loss()(predicted_yields, yields)

            loss.backward()
            optimizer.step()
            batch_train_loss.append(loss.cpu().detach().numpy())

        # Finding the average train loss.
        logger.debug('Mean Train Loss at epoch %d is %.4f', epoch, np.mean(batch_train_loss))

    # Evaluating on test set.
    logger.debug('Training Finished. Starting evaluation on test set.')

    model.eval()
    all_predicted_yields = []
    all_true_yields = []
    with torch.no_grad():
        for i, (halide_features, halide_adj_matrices, ligand_features, ligand_adj_matrices,
                base_features, base_adj_matrices, additive_features,
                additive_adj_matrices, yields) in enumerate(test_dataloader):

            halide_features = halide_features.to(device)
            halide_adj_matrices = halide_adj_matrices.to(device)
            ligand_features = ligand_features.to(device)
            ligand_adj_matrices = ligand_adj_matrices.to(device)
            base_features = base_features.to(device)
            base_adj_matrices = base_adj_matrices.to(device)
            additive_features = additive_features.to(device)
            additive_adj_matrices = additive_adj_matrices.to(device)
            yields = yields.to(device)

            predicted_yields = model(halide_adj_matrices, halide_features, ligand_adj_matrices,
                                     ligand_features, base_adj_matrices, base_features, additive_adj_matrices,
                                     additive_features)

            loss = torch.nn.L1Loss()(predicted_yields, yields)

            all_predicted_yields = all_predicted_yields + list(predicted_yields.cpu().detach().numpy())
            all_true_yields = all_true_yields + list(yields.cpu().detach().numpy())

    # Saving out the models and predictions.
    final_model_save_path = os.path.join(save_path, 'model')
    torch.save(model.state_dict(), final_model_save_path)

    preds_save_path = os.path.join(save_path, 'preds.pickle')
    preds['predicted_yield'] = all_predicted_yields
    preds['true_yield'] = all_true_yields
    preds['loss'] = np.abs(preds['predicted_yield'] - preds['true_yield'])

    logger.debug('Test MAE is %.4f', np.mean(preds['loss']))
    # logger.debug('Test Median Loss is %.4f', np.median(preds['loss']))
    # logger.debug("Test 'Chemist' MAE is %.4f", np.mean([0 if x <= 10 else x for x in preds['loss']]))

    preds.to_pickle(preds_save_path)

if __name__ == '__main__':
    main()
