'''
The MPNN to classify the odor classes of compounds.
'''


import torch
import torch.nn as nn
import sys
from collections import OrderedDict
sys.path.append('../MPNN/')

from mpnn import Big_MPNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Fragrance_MPNN(nn.Module):
    def __init__(self, message_size, message_passes, atom_list,
                 pretrained_mpnn_path, number_of_labels):
        super(Fragrance_MPNN, self).__init__()

        self.message_size = message_size
        self.message_passes = message_passes
        self.atom_list = atom_list
        self.pretrained_mpnn_path = pretrained_mpnn_path
        self.number_of_labels = number_of_labels

        self.mpnn = Big_MPNN(self.message_size, self.message_passes, self.atom_list)
        mpnn_trained_state_dict = self.gen_states(self.pretrained_mpnn_path)
        self.mpnn.load_state_dict(mpnn_trained_state_dict)

        # Turning off the params.
        for param in self.mpnn.parameters():
            param.requires_grad = False

        self.fragrance_pedictor = nn.Sequential(
            nn.Linear(self.message_size, self.message_size),
            nn.ReLU(),
            nn.Linear(self.message_size, self.number_of_labels)
        )

    def forward(self, molecule_matricies, molecule_features):
        molecule_embedding = self.mpnn(molecule_matricies, molecule_features)
        molecule_embedding = torch.sum(molecule_embedding, dim=1)

        # Predicting the toxicity
        predicted_fragrance_classes = self.fragrance_pedictor(molecule_embedding)

        return nn.Sigmoid()(predicted_fragrance_classes)

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
