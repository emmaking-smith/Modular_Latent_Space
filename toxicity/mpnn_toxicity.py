'''
The MPNN to predict the toxicity of compounds.
'''

import torch
import torch.nn as nn
import sys
from collections import OrderedDict
sys.path.append('../MPNN/')

from mpnn import Big_MPNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Toxicity_MPNN(nn.Module):
    def __init__(self, message_size, message_passes, atom_list, pretrained_mpnn_path, longest_molecule):
        super(Toxicity_MPNN, self).__init__()

        self.message_size = message_size
        self.message_passes = message_passes
        self.atom_list = atom_list
        self.pretrained_mpnn_path = pretrained_mpnn_path
        self.longest_molecule = longest_molecule

        self.mpnn = Big_MPNN(self.message_size, self.message_passes, self.atom_list)
        mpnn_trained_state_dict = self.gen_states(self.pretrained_mpnn_path)
        self.mpnn.load_state_dict(mpnn_trained_state_dict)

        # Turning off the params.
        for param in self.mpnn.parameters():
            param.requires_grad = False

        self.toxcicity_pedictor = nn.Sequential(
            nn.Linear(self.message_size, self.message_size),
            nn.ReLU(),
            nn.Linear(self.message_size, 1)
        )

        self.compress_mol = nn.Sequential(
            nn.Linear(self.longest_molecule, self.longest_molecule, bias=False),
            nn.ReLU(),
            nn.Linear(self.longest_molecule, 1, bias=False)
        )

    def forward(self, molecule_matricies, molecule_features):
        molecule_embedding = self.mpnn(molecule_matricies, molecule_features)
        #molecule_embedding = self.compress_mol(molecule_embedding.swapaxes(1,-1)).swapaxes(1,-1)
        molecule_embedding = torch.sum(molecule_embedding, dim=1)
        # Predicting the toxicity
        predicted_toxicity = self.toxcicity_pedictor(molecule_embedding)

        return predicted_toxicity.view([-1])

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

