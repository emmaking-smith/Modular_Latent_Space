'''
The MPNN to predict yeilds of Suzuki reactions
'''

import torch
import torch.nn as nn
import sys
from collections import OrderedDict
sys.path.append('../MPNN/')

from mpnn import Big_MPNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Suzuki_MPNN(nn.Module):
    def __init__(self, message_size, message_passes, atom_list, pretrained_mpnn_path):
        super(Suzuki_MPNN, self).__init__()

        self.message_size = message_size
        self.message_passes = message_passes
        self.atom_list = atom_list
        self.pretrained_mpnn_path = pretrained_mpnn_path

        self.mpnn = Big_MPNN(self.message_size, self.message_passes, self.atom_list)
        mpnn_trained_state_dict = self.gen_states(self.pretrained_mpnn_path)
        self.mpnn.load_state_dict(mpnn_trained_state_dict)

        # Turning off the params.
        for param in self.mpnn.parameters():
            param.requires_grad = False

        self.yield_predictor = nn.Sequential(
            nn.Linear(self.message_size * 4, self.message_size * 4),
            nn.ReLU(),
            nn.Linear(self.message_size * 4, 1)
        )

    def forward(self, boronic_acid_matrices, boronic_acid_features, halide_matrices, halide_features,
                catalyst_matrices, catalyst_features, ligand_matrices, ligand_features):
        boronic_acid_embedding = self.mpnn(boronic_acid_matrices, boronic_acid_features)
        halide_embedding = self.mpnn(halide_matrices, halide_features)
        catalyst_embedding = self.mpnn(catalyst_matrices, catalyst_features)
        ligand_embedding = self.mpnn(ligand_matrices, ligand_features)

        # Sum along columns to yield a batch x message_size vector for each embedded molecule.
        boronic_acid_embedding = torch.sum(boronic_acid_embedding, dim=1)
        halide_embedding = torch.sum(halide_embedding, dim=1)
        catalyst_embedding = torch.sum(catalyst_embedding, dim=1)
        ligand_embedding = torch.sum(ligand_embedding, dim=1)

        # Creating the batch x 4 * message_size reaction vector containing all embedded molecules.
        reaction_vector = torch.cat([boronic_acid_embedding, halide_embedding,
                                     catalyst_embedding, ligand_embedding], dim=1)

        # Predicting the yield
        predicted_yield = self.yield_predictor(reaction_vector)


        predicted_yield = nn.ReLU()(predicted_yield)
        # predicted_yield = torch.abs(predicted_yield)
        predicted_yield = predicted_yield.view([-1])

        return predicted_yield

    def gen_states(self, pretrained_model_path):
        '''
        Loading in the states for the mpnn,
        allowing for new modules to be tacked on.
        '''
        new_state_dict = OrderedDict()
        state_dict = torch.load(pretrained_model_path, map_location=device)
        for key, value in state_dict.items():
            if 'mpnn' in key:
                new_key = key.split('mpnn.')[1]
                new_state_dict[new_key] = value
        return new_state_dict

