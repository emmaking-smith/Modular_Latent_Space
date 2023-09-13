'''
Suzuki yield prediction with Big MPNN.
'''

import numpy as np
import pandas as pd

from suzuki_splits import load_parsed_data, ranking_nucleophiles_and_electrophiles, split_on_reactant, \
    clean_ligands, find_catalysts_ligands, find_reactants, reformat_ligs_and_cats, split_on_multiple_electrophiles, \
    split_on_cat_lig, ranking_catalysts_and_ligands
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import os
from rdkit import Chem
from pathlib import Path
from suzuki_pytorch_dataset import Suzuki_USPTO_Rxns, collate_fn
from suzuki_yield_mpnn import Suzuki_MPNN

class Initialization:
    def __init__(self):
        pass

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--paths_to_parsed_suzuki_data', type=list, help='The paths to the parsed Suzuki text files.',
                            default=['suzuki_from_arom_USPTO_parsed_het.txt', 'suzuki_from_arom_USPTO_parsed_homo.txt'])
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--message_size', type=int, default=128)
        parser.add_argument('--message_passes', type=int, default=3)
        parser.add_argument('--pretrained_mpnn_path', type=str, default='../MPNN/big_mpnn_no_delocalised_no_unknown_model')
        parser.add_argument('--save_path', '-s', type=str)
        parser.add_argument('--split', type=str, help='Options are: nucleophile, electrophile, catalyst, ligand.')
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
    def __init__(self, rxn_dict):
        self.rxn_dict = rxn_dict

    def find_longest_boronic_acid(self):
        boronic_acids = [find_reactants(self.rxn_dict[x]['rx'])[0] for x in list(self.rxn_dict.keys())]
        boronic_acid_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in boronic_acids]
        return max(boronic_acid_lengths)

    def find_longest_halide(self):
        halides = [find_reactants(self.rxn_dict[x]['rx'])[1] for x in list(self.rxn_dict.keys())]
        halides_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in halides]
        return max(halides_lengths)

    def find_longest_catalyst(self):
        catalysts = [find_catalysts_ligands(self.rxn_dict[x])[0] for x in list(self.rxn_dict.keys())]
        catalysts = reformat_ligs_and_cats(catalysts)
        catalysts_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in catalysts]
        return max(catalysts_lengths)

    def find_longest_ligand(self):
        ligands = [find_catalysts_ligands(self.rxn_dict[x])[1] for x in list(self.rxn_dict.keys())]
        ligands = clean_ligands(ligands)
        ligands = reformat_ligs_and_cats(ligands)
        ligands_lengths = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in ligands]
        return max(ligands_lengths)

def main():
    args = Initialization().init_args()
    paths_to_suzuki_data = args.paths_to_parsed_suzuki_data
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    message_size = args.message_size
    message_passes = args.message_passes
    pretrained_mpnn_path = args.pretrained_mpnn_path
    save_path = args.save_path
    split = args.split

    atom_list = [6, 8, 7, 9, 17, 16, 15, 5, 29, 35, 14, 53, 30, 26, 27, 28, 48, 44, 42, 25, 47, 34, 46, 78, 50, 74,
                 11, 3, 19, 23, 13, 79, 45, 82, 75, 77, 76, 51, 24, 33, 22, 32, 80, 92, 31, 63, 52, 12, 40, 65, 83,
                 49, 64, 20, 66, 57, 60, 62, 39, 56, 68, 59, 58, 70, 55, 38, 41, 37, 73, 67, 21, 81, 71, 72, 90, 69,
                 43, 4, 93, 94, 54, 95, 2, 10, 18, 98, 36, 96, 97, 91]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = Initialization().logging_setup()
    logger.debug('Our device is %s', device)

    suzuki_rxns = load_parsed_data(paths_to_suzuki_data[0], start_index=0)
    suzuki_rxns_2 = load_parsed_data(paths_to_suzuki_data[1], start_index=len(suzuki_rxns))
    suzuki_rxns.update(suzuki_rxns_2)

    # Deleting the rxns with super long catalysts.
    del suzuki_rxns[3818], suzuki_rxns[420], suzuki_rxns[2922]

    longest_boronic_acid = Find_Longest_Molecules(suzuki_rxns).find_longest_boronic_acid() + 1
    longest_halide = Find_Longest_Molecules(suzuki_rxns).find_longest_halide() + 1
    longest_catalyst = Find_Longest_Molecules(suzuki_rxns).find_longest_catalyst() + 1
    longest_ligand = Find_Longest_Molecules(suzuki_rxns).find_longest_ligand() + 1

    logger.debug('Data Loaded.')

    # Dataframe splits.
    if split in ['catalyst', 'ligand']:
        most_common_catalysts, most_common_ligands = ranking_catalysts_and_ligands(suzuki_rxns)
        if split == 'catalyst':
            train_val_dict, test_dict = split_on_cat_lig(suzuki_rxns, most_common_catalysts[
                0])  # if using multiple electrophile splits it's 0:35, and 35:
            train_dict, val_dict = split_on_cat_lig(train_val_dict, most_common_catalysts[1])
        else:
            train_val_dict, test_dict = split_on_cat_lig(suzuki_rxns, most_common_ligands[
                0])  # if using multiple electrophile splits it's 0:35, and 35:
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

    train_dataset = Suzuki_USPTO_Rxns(train_dict, longest_boronic_acid, longest_halide, longest_catalyst, longest_ligand)
    val_dataset = Suzuki_USPTO_Rxns(val_dict, longest_boronic_acid, longest_halide, longest_catalyst, longest_ligand)
    test_dataset = Suzuki_USPTO_Rxns(test_dict, longest_boronic_acid, longest_halide, longest_catalyst, longest_ligand)

    # Define optimizer and model.
    model = Suzuki_MPNN(message_size, message_passes, atom_list, pretrained_mpnn_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.debug('Model defined. Message Size: %d, Message Passes: %d, Atom List: %s, Pretrained MPNN Path: %s',
                 message_size, message_passes, atom_list, pretrained_mpnn_path)

    # Define the dataloaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    del train_dataset, val_dataset, test_dataset, atom_list

    overall_test_loss = []

    # Train Loop
    model.to(device).float()

    for epoch in range(epochs):
        batch_train_loss = []
        batch_val_loss = []
        model.train()
        for i, (halide_features, halide_adj_matrices, boronic_acid_features,
                boronic_acid_adj_matrices, catalyst_features, catalyst_adj_matrices,
                ligand_features, ligand_adj_matrices, yields) in enumerate(train_dataloader):

            halide_features = halide_features.to(device)
            halide_adj_matrices = halide_adj_matrices.to(device)
            boronic_acid_features = boronic_acid_features.to(device)
            boronic_acid_adj_matrices = boronic_acid_adj_matrices.to(device)
            catalyst_features = catalyst_features.to(device)
            catalyst_adj_matrices = catalyst_adj_matrices.to(device)
            ligand_features = ligand_features.to(device)
            ligand_adj_matrices = ligand_adj_matrices.to(device)
            yields = yields.to(device)

            optimizer.zero_grad()

            predicted_yields = model(boronic_acid_adj_matrices, boronic_acid_features, halide_adj_matrices,
                                     halide_features, catalyst_adj_matrices, catalyst_features, ligand_adj_matrices,
                                     ligand_features)

            loss = torch.nn.L1Loss()(predicted_yields, yields)

            loss.backward()
            optimizer.step()
            batch_train_loss.append(loss.cpu().detach().numpy())

        # Finding the average train loss.
        logger.debug('Mean Train Loss at epoch %d is %.4f', epoch, np.mean(batch_train_loss))

        # Check on Val set
        model.eval()
        all_val_predicted_yields = []
        all_val_true_yields = []
        with torch.no_grad():
            for i, (halide_features, halide_adj_matrices, boronic_acid_features,
                    boronic_acid_adj_matrices, catalyst_features, catalyst_adj_matrices,
                    ligand_features, ligand_adj_matrices, yields) in enumerate(val_dataloader):
                halide_features = halide_features.to(device)
                halide_adj_matrices = halide_adj_matrices.to(device)
                boronic_acid_features = boronic_acid_features.to(device)
                boronic_acid_adj_matrices = boronic_acid_adj_matrices.to(device)
                catalyst_features = catalyst_features.to(device)
                catalyst_adj_matrices = catalyst_adj_matrices.to(device)
                ligand_features = ligand_features.to(device)
                ligand_adj_matrices = ligand_adj_matrices.to(device)
                yields = yields.to(device)

                predicted_yields = model(boronic_acid_adj_matrices, boronic_acid_features, halide_adj_matrices,
                                         halide_features, catalyst_adj_matrices, catalyst_features, ligand_adj_matrices,
                                         ligand_features)

                loss = torch.nn.L1Loss()(predicted_yields, yields)
                batch_val_loss.append(loss.cpu().detach().numpy())

                if epoch == epochs - 1:
                    all_val_predicted_yields = all_val_predicted_yields + list(predicted_yields.cpu().detach().numpy())
                    all_val_true_yields = all_val_true_yields + list(yields.cpu().detach().numpy())

        # Finding the average batch train loss for the epoch.
        logger.debug('Mean Val Loss at epoch %d is %.4f', epoch, np.mean(batch_val_loss))
        if epoch == epochs - 1:
            logger.debug('Overall MAE Val Loss at final epoch is %.4f', np.mean(np.abs(np.array(all_val_predicted_yields) - np.array(all_val_true_yields))))
            logger.debug('Overall Median Val Loss at final epoch is %.4f', np.median(np.abs(np.array(all_val_predicted_yields) - np.array(all_val_true_yields))))
            chemist_mae = []
            for pred, true in zip(all_val_predicted_yields, all_val_true_yields):
                if np.abs(pred - true) <= 10:
                    chemist_mae.append(0)
                else:
                    chemist_mae.append(np.abs(pred - true))
            logger.debug("Overall 'Chemist' MAE Val Loss at final epoch is %.4f", np.mean(chemist_mae))

    # Evaluating on test set.
    logger.debug('Training Finished. Starting evaluation on test set.')

    model.eval()
    preds = pd.DataFrame()
    all_predicted_yields = []
    all_true_yields = []
    with torch.no_grad():
        for i, (halide_features, halide_adj_matrices, boronic_acid_features,
                    boronic_acid_adj_matrices, catalyst_features, catalyst_adj_matrices,
                    ligand_features, ligand_adj_matrices, yields) in enumerate(test_dataloader):

            halide_features = halide_features.to(device)
            halide_adj_matrices = halide_adj_matrices.to(device)
            boronic_acid_features = boronic_acid_features.to(device)
            boronic_acid_adj_matrices = boronic_acid_adj_matrices.to(device)
            catalyst_features = catalyst_features.to(device)
            catalyst_adj_matrices = catalyst_adj_matrices.to(device)
            ligand_features = ligand_features.to(device)
            ligand_adj_matrices = ligand_adj_matrices.to(device)
            yields = yields.to(device)

            predicted_yields = model(boronic_acid_adj_matrices, boronic_acid_features, halide_adj_matrices,
                                     halide_features, catalyst_adj_matrices, catalyst_features, ligand_adj_matrices,
                                     ligand_features)

            loss = torch.nn.L1Loss()(predicted_yields, yields)
            overall_test_loss.append(loss.cpu().detach().numpy())

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
    logger.debug('Test Median Loss is %.4f', np.median(preds['loss']))
    logger.debug("Test 'Chemist' MAE is %.4f", np.mean([0 if x <= 10 else x for x in preds['loss']]))
    preds.to_pickle(preds_save_path)

if __name__ == '__main__':
    main()
