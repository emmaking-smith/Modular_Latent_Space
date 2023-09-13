'''
A MPNN based on the LSF work I did.
'''

import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

class Big_MPNN(nn.Module):
    def __init__(self, message_size, message_passes, ranked_unique_atoms):
        super(Big_MPNN, self).__init__()

        self.message_passes = message_passes
        self.message_size = message_size
        self.ranked_unique_atoms = ranked_unique_atoms
        self.top_4_unique_atoms = ranked_unique_atoms[0:4] + [0]
        self.bond_types = ['Single', 'Double', 'Triple', 'Quadruple',
                           'Aromatic', 'Pi', 'Universal']

        # The bond-specific message functions.
        self.message_func = nn.ModuleDict()

        for bond in self.bond_types:
            self.message_func[bond] = nn.Sequential(
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
            )

        # The atom-specific GRU update function for the non-universal and universal nodes.
        self.update_func = nn.ModuleDict()
        self.update_func_universal = nn.ModuleDict()

        for atom in self.top_4_unique_atoms:
            self.update_func[str(atom)] = nn.GRUCell(self.message_size, self.message_size)
            self.update_func_universal[str(atom)] = nn.GRUCell(self.message_size, self.message_size)

        # The catchall GRU update function for the non-universal and universal nodes.
        self.update_func_catchall = nn.GRUCell(self.message_size, self.message_size)
        self.update_func_catchall_universal = nn.GRUCell(self.message_size, self.message_size)

    def edge_propogation(self, g, h_t, bond_type, bond_number):
        '''
        Propogation of a message through a single bond type. Bond number is the index in which
        that bond is listed in self.bond_types.
        '''
        h_t_bond_type = self.message_func[bond_type](h_t)
        m_bond_type = torch.bmm(g[:, bond_number], h_t_bond_type)
        return m_bond_type

    def forward(self, g, h):
        batch_size = g.size()[0]

        # Padding the atomic representations to some higher dimension, d = message size.
        h_t = torch.cat([h, torch.zeros(h.size()[0], h.size()[1], self.message_size - h.size()[2]).type_as(h.data)],
                        2)

        # Finding the order of atoms for the input.
        atom_numbers = h[:, :, 0].view([-1])

        # Message Passing Loop
        for i in range(self.message_passes):
            # Running the padded atomic information through edge propogation. Note that the universal node
            # is treated separately.
            m_non_universal = sum(
                [self.edge_propogation(g, h_t, bond, i) for i, bond in enumerate(self.bond_types[0:-1])])
            m_universal = self.edge_propogation(g, h_t, 'Universal', len(self.bond_types) - 1)

            h_no_batches = h_t.view([-1, h_t.size()[2]])
            m_no_batches = m_non_universal.view([-1, m_non_universal.size()[2]])
            m_uni_no_batches = m_universal.view([-1, m_non_universal.size()[2]])

            gru_output = torch.empty_like(h_no_batches)
            gru_uni_output = torch.empty_like(h_no_batches)

            for atom_type in torch.unique(atom_numbers):
                # Find the rows that correspond to that atom type.
                h_atom_type_subset = h_no_batches.index_select(0, torch.where(atom_numbers == atom_type)[0])
                m_atom_type_subset = m_no_batches.index_select(0, torch.where(atom_numbers == atom_type)[0])
                m_uni_atom_type_subset = m_uni_no_batches.index_select(0, torch.where(atom_numbers == atom_type)[0])

                # If that atom type is in the top 4 atoms, run through the specific GRUs.
                if atom_type in self.top_4_unique_atoms:
                    idx = str(int(atom_type.detach().cpu().numpy()))

                    gru_atom_type_subset = self.update_func[idx](h_atom_type_subset, m_atom_type_subset)
                    gru_uni_atom_type_subset = self.update_func_universal[idx](h_atom_type_subset,
                                                                               m_uni_atom_type_subset)

                    gru_output[torch.where(atom_numbers == atom_type)[0]] = gru_atom_type_subset
                    gru_uni_output[torch.where(atom_numbers == atom_type)[0]] = gru_uni_atom_type_subset

                # Else run that atom through the catchall GRU.
                else:
                    gru_atom_type_subset = self.update_func_catchall(h_atom_type_subset, m_atom_type_subset)
                    gru_uni_atom_type_subset = self.update_func_catchall_universal(h_atom_type_subset,
                                                                                   m_uni_atom_type_subset)

                    gru_output[torch.where(atom_numbers == atom_type)[0]] = gru_atom_type_subset
                    gru_uni_output[torch.where(atom_numbers == atom_type)[0]] = gru_uni_atom_type_subset

            # Adding the universal bond hidden states to the hidden states from the
            # single/double/triple/aromatic bond states.
            gru_total = gru_output + gru_uni_output

            # Putting the batches back in.
            h_t = gru_total.view([batch_size, h_t.size()[1], h_t.size()[2]])

        return h_t
