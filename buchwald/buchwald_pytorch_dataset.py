'''
Creating the Dataset for the DataLoader

Using Dative Bonds (Corresponding to "Pi" in CCDC dataset).
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
import networkx as nx

metals = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
          71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
          97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]

def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)

def set_dative_bonds(mol, fromAtoms=(7,8,13,15,16)):
    """ convert some bonds to dative
    fromAtoms = N, O, Al, P, S
    Replaces some single bonds between metals and atoms with atomic numbers in fromAtoms
    with dative bonds. The replacement is only done if the atom has "too many" bonds.

    Returns the modified molecule.

    """
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in fromAtoms and \
               nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
               rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
    return rwmol

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

class Buchwald_Rxns(Dataset):
    def __init__(self, df, longest_halide, longest_ligand, longest_base, longest_additive):
        super(Buchwald_Rxns, self).__init__()

        self.df = df
        self.longest_halide = longest_halide
        self.longest_ligand = longest_ligand
        self.longest_base = longest_base
        self.longest_additive = longest_additive

    def __len__(self):
        return len(self.df)

    def is_metal(self, atom):
        metallic = False
        if atom.GetAtomicNum() in metals:
            metallic = True
        return metallic

    def make_graph(self, canonical_smiles):

        m = Chem.MolFromSmiles(canonical_smiles, sanitize=False)

        # Adding in dative bonds, in any exist.
        m = Chem.MolFromSmiles(Chem.MolToSmiles(set_dative_bonds(m)))

        # RDKit nonsense for finding the features of our molecule.
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        feats = factory.GetFeaturesForMol(m)

        g = nx.Graph()

        # Create nodes for each bond type.
        for i, atom in enumerate(m.GetAtoms()):
            # Note that RDKit goes in order of atom index, which is constant.

            g.add_node(i, a_type=atom.GetSymbol(), a_num=atom.GetAtomicNum(), acceptor=0, donor=0,
                       aromatic=atom.GetIsAromatic(), hybridization=atom.GetHybridization(),
                       num_h=atom.GetTotalNumHs(), chiral=atom.GetChiralTag(), cyclic=atom.IsInRing(),
                       formal_charge=atom.GetFormalCharge(), metal=self.is_metal(atom))
        # Here 'Donor' and 'Acceptor' refer to hydrogen bond donors or acceptors.
        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1

        g_single = g.copy()
        g_double = g.copy()
        g_triple = g.copy()
        g_quadruple = g.copy()
        g_aromatic = g.copy()
        g_dative = g.copy()

        # Read Edges - no distances, split g based on bond type
        for i in range(len(m.GetAtoms())):
            for j in range(len(m.GetAtoms())):
                e_ij = m.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    if e_ij.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        # Add bond to single.
                        g_single.add_edge(i, j, b_type=e_ij.GetBondType())

                    elif e_ij.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        # Add bond to double.
                        g_double.add_edge(i, j, b_type=e_ij.GetBondType())

                    elif e_ij.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                        # Add bond to triple.
                        g_triple.add_edge(i, j, b_type=e_ij.GetBondType())

                    elif e_ij.GetBondType() == Chem.rdchem.BondType.QUADRUPLE:
                        # Add bond to quadruple.
                        g_quadruple.add_edge(i, j, b_type=e_ij.GetBondType())

                    elif e_ij.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        # Add bond to aromatic.
                        g_aromatic.add_edge(i, j, b_type=e_ij.GetBondType())

                    elif e_ij.GetBondType() == Chem.rdchem.BondType.DATIVE:
                        # Add bond to dative.
                        g_dative.add_edge(i, j, b_type=e_ij.GetBondType())
                else:
                    pass
        return g_single, g_double, g_triple, g_quadruple, g_aromatic, g_dative

    def make_atom_features(self, graph, longest_molecule):
        h = []
        # Generating the hidden features vector for each atom in the molecule
        for n, d in graph.nodes_iter(data=True):
            h_t = []
            # Atomic number
            h_t.append(d['a_num'])
            # Acceptor
            h_t.append(d['acceptor'])
            # Donor
            h_t.append(d['donor'])
            # Chiral
            h_t.append(int(d['chiral']))
            # Implicit number of hydrogens
            h_t.append(d['num_h'])
            # Cyclic
            h_t.append(int(d['cyclic']))
            # Metallic
            h_t.append(int(d['metal']))
            # Formal Charge
            h_t.append(d['formal_charge'])
            # Universal node one-hot.
            h_t.append(0)
            h.append(h_t)
        # Appending the universal node as the n+1th node.
        h.append([0] * (len(h[0]) - 1) + [1])
        # Padding h with 0's to match longest molecule size.
        if len(h) < longest_molecule:
            atom_discrepancy = longest_molecule - len(h)
            for i in range(atom_discrepancy):
                h.append([0] * (len(h[0])))
        return h  # h now padded to longest molecule size.

    def graph_padding(self, graph, longest_molecule):

        g_length = len(graph)
        if g_length < longest_molecule:
            padding = longest_molecule - g_length
            # Forming the zero column we will add. Note that we need to make it
            # an array with dimensions len(graph) x 1
            col_padding = np.zeros([g_length, 1])

            # Appending the column padding vector to the graph padding times.
            padding_counter = 1
            while padding_counter <= padding:
                graph = np.append(graph, col_padding, axis=1)
                padding_counter += 1

            # Repeating the above for the rows. Note that the graph dimensions
            # should now be M x N where N is the longest molecule size.
            row_padding = np.zeros([1, longest_molecule])

            padding_counter = 1
            while padding_counter <= padding:
                graph = np.append(graph, row_padding, axis=0)
                padding_counter += 1
        else:
            pass
        return graph

    def universal_node_graph(self, molecule_size, longest_molecule):

        g_universal = np.zeros(longest_molecule * longest_molecule).reshape((longest_molecule, longest_molecule))
        for i in range(molecule_size):
            g_universal[i, molecule_size] = 1
            g_universal[molecule_size, i] = 1

        return g_universal

    def single_molecule_matrices_and_features_workflow(self, smiles, longest_molecule):
        '''
        Taking a molecule from smiles to all adjacency matrices and features vector.
        '''
        if len(smiles) == 0:
            return np.zeros([longest_molecule, 9]), \
                   np.array([np.zeros([longest_molecule * 7, longest_molecule])]).reshape([-1, longest_molecule, longest_molecule])
        else:
            nha = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()

            # Networkx graphs: g_dative will be going through Pi bond message pass.
            g_single, g_double, g_triple, g_quadruple, g_aromatic, g_dative = self.make_graph(smiles)

            # Feature vector
            features = self.make_atom_features(g_single, longest_molecule)

            g = np.concatenate((nx.to_numpy_matrix(g_single), nx.to_numpy_matrix(g_double), nx.to_numpy_matrix(g_triple),
                                nx.to_numpy_matrix(g_quadruple), nx.to_numpy_matrix(g_aromatic),
                                nx.to_numpy_matrix(g_dative)), axis=0)

            g = np.array(g).reshape([-1, nx.to_numpy_matrix(g_single).shape[0], nx.to_numpy_matrix(g_single).shape[1]])

            # Adding the graph padding.
            g_pad = np.zeros([g.shape[0], longest_molecule, longest_molecule])
            for i in range(g.shape[0]):  # g.shape[0] = number of bond types.
                g_pad[i] = self.graph_padding(g[i], longest_molecule)

            # Make universal node graph.
            g_universal = self.universal_node_graph(nha, longest_molecule)
            g = np.concatenate((g_pad, g_universal.reshape([1, longest_molecule, longest_molecule])), axis=0)

            return features, g

    def __getitem__(self, index):
        # Get the canonical smiles of aryl halide, ligand, base, and additive.
        # Note: Amine stays the same.
        # print(f'############ index {index} #################')
        # print('halide', self.df.loc[index, 'aryl_halide_smiles'])
        # print('ligand', self.df.loc[index, 'ligand_smiles'])
        # print('base', self.df.loc[index, 'base_smiles'])
        # print('additive', self.df.loc[index, 'additive_smiles'])
        # print('----------------------------------------------')
        # print()

        halide = canonicalize_smiles(self.df.loc[index, 'aryl_halide_smiles'])
        ligand = canonicalize_smiles(self.df.loc[index, 'ligand_smiles'])
        base = canonicalize_smiles(self.df.loc[index, 'base_smiles'])
        additive = canonicalize_smiles(self.df.loc[index, 'additive_smiles'])

        # Creating the features vectors.
        halide_features, halide_adj_matrices = self.single_molecule_matrices_and_features_workflow(halide,
                                                                                                               self.longest_halide)
        ligand_features, ligand_adj_matrices = self.single_molecule_matrices_and_features_workflow(ligand,
                                                                                                   self.longest_ligand)
        base_features, base_adj_matrices = self.single_molecule_matrices_and_features_workflow(base, self.longest_base)

        additive_features, additive_adj_matrices = self.single_molecule_matrices_and_features_workflow(additive, self.longest_additive)

        yields = self.df.loc[index, 'yield']

        halide_features = torch.tensor(halide_features).float()
        halide_adj_matrices = torch.tensor(halide_adj_matrices).float()

        ligand_features = torch.tensor(ligand_features).float()
        ligand_adj_matrices = torch.tensor(ligand_adj_matrices).float()

        base_features = torch.tensor(base_features).float()
        base_adj_matrices = torch.tensor(base_adj_matrices).float()

        additive_features = torch.tensor(additive_features).float()
        additive_adj_matrices = torch.tensor(additive_adj_matrices).float()

        yields = torch.tensor([yields])
        return (halide_features, halide_adj_matrices, ligand_features, ligand_adj_matrices,
                base_features, base_adj_matrices, additive_features, additive_adj_matrices), (yields)

def collate_fn(batch):
    batch_size = len(batch)
    halide_features, halide_adj_matrices, ligand_features, ligand_adj_matrices, base_features, base_adj_matrices, additive_features, additive_adj_matrices = batch[0][0]
    yields = batch[0][1]

    longest_halide = halide_adj_matrices.size()[-1]
    longest_ligand = ligand_adj_matrices.size()[-1]
    longest_base = base_adj_matrices.size()[-1]
    longest_additive = additive_adj_matrices.size()[-1]

    # Concatenate all together.
    for i in range(1, batch_size):
        halide_features = torch.cat([halide_features, batch[i][0][0]], 0)
        halide_adj_matrices = torch.cat([halide_adj_matrices, batch[i][0][1]], 0)

        ligand_features = torch.cat([ligand_features, batch[i][0][2]], 0)
        ligand_adj_matrices = torch.cat([ligand_adj_matrices, batch[i][0][3]], 0)

        base_features = torch.cat([base_features, batch[i][0][4]], 0)
        base_adj_matrices = torch.cat([base_adj_matrices, batch[i][0][5]], 0)

        additive_features = torch.cat([additive_features, batch[i][0][6]], 0)
        additive_adj_matrices = torch.cat([additive_adj_matrices, batch[i][0][7]], 0)

        yields = torch.cat([yields, batch[i][1]], 0)

    # Reshape as desired.
    halide_features = halide_features.view([batch_size, longest_halide, -1])
    halide_adj_matrices = halide_adj_matrices.view([batch_size, -1, longest_halide, longest_halide])

    ligand_features = ligand_features.view([batch_size, longest_ligand, -1])
    ligand_adj_matrices = ligand_adj_matrices.view([batch_size, -1, longest_ligand, longest_ligand])

    base_features = base_features.view([batch_size, longest_base, -1])
    base_adj_matrices = base_adj_matrices.view([batch_size, -1, longest_base, longest_base])

    additive_features = additive_features.view([batch_size, longest_additive, -1])
    additive_adj_matrices = additive_adj_matrices.view([batch_size, -1, longest_additive, longest_additive])

    yields = yields.view([-1])
    return halide_features, halide_adj_matrices, ligand_features, ligand_adj_matrices, base_features, base_adj_matrices, additive_features, additive_adj_matrices, yields