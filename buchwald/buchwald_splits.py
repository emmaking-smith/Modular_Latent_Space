'''
Split the dataset based on each halides, ligands, addtives, and bases.
'''


from rdkit import Chem
from collections import Counter
import pandas as pd

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def remove_nans(df):
    df.loc[pd.isna(df['aryl_halide_smiles']) == True, ['aryl_halide_smiles']] = ''
    df.loc[pd.isna(df['ligand_smiles']) == True, ['ligand_smiles']] = ''
    df.loc[pd.isna(df['base_smiles']) == True, ['base_smiles']] = ''
    df.loc[pd.isna(df['additive_smiles']) == True, ['additive_smiles']] = ''
    return df

def find_halide(df_row):
    return canonicalize_smiles(df_row['aryl_halide_smiles'])

def find_ligand(df_row):
    return canonicalize_smiles(df_row['ligand_smiles'])

def find_base(df_row):
    return canonicalize_smiles(df_row['base_smiles'])

def find_additive(df_row):
    return canonicalize_smiles(df_row['additive_smiles'])

def find_mol_ranking(df):
    '''
    Finding most common electrophiles and nucleophiles used in rxns.
    '''
    halides = [find_halide(df.iloc[x]) for x in list(df.index)]
    ligands = [find_ligand(df.iloc[x]) for x in list(df.index)]
    bases = [find_base(df.iloc[x]) for x in list(df.index)]
    additives = [find_additive(df.iloc[x]) for x in list(df.index)]

    ranked_halides = [Counter(halides).most_common()[x][0] for x in range(len(Counter(halides).most_common()))]
    ranked_ligands = [Counter(ligands).most_common()[x][0] for x in range(len(Counter(ligands).most_common()))]
    ranked_bases = [Counter(bases).most_common()[x][0] for x in range(len(Counter(bases).most_common()))]
    ranked_additives = [Counter(additives).most_common()[x][0] for x in range(len(Counter(additives).most_common()))]

    return ranked_halides, ranked_ligands, ranked_bases, ranked_additives

def split_on_molecule(df, canon_smiles, reagent_type):
    '''
    Given a reactant to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''

    assert reagent_type in ['ligand', 'base']

    excludes_mol_df = df.copy()
    includes_mol_df = df.copy()

    includes_mol_idxs = []
    for i in list(df.index):
        if reagent_type == 'ligand':
            reactants = find_ligand(df.iloc[i])
        else:
            reactants = find_base(df.iloc[i])

        if canon_smiles == reactants:
            includes_mol_idxs.append(i)

    excludes_react_idxs = list(set(list(df.index)) - set(includes_mol_idxs))

    includes_mol_df = includes_mol_df.drop(excludes_react_idxs)
    excludes_mol_df = excludes_mol_df.drop(includes_mol_idxs)

    return excludes_mol_df, includes_mol_df

def split_on_molecules(df, list_of_canon_smiles, reagent_type):
    '''
    Given a reactant to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''

    assert reagent_type in ['halide', 'additive']

    excludes_mol_df = df.copy()
    includes_mol_df = df.copy()

    includes_mol_idxs = []
    for i in list(df.index):
        if reagent_type == 'halide':
            reactants = find_halide(df.iloc[i])
        else:
            reactants = find_additive(df.iloc[i])

        for smiles in list_of_canon_smiles:
            if smiles == '':
                if smiles == reactants:
                    includes_mol_idxs.append(i)
            else:
                if smiles in reactants:
                    includes_mol_idxs.append(i)

    excludes_react_idxs = list(set(list(df.index)) - set(includes_mol_idxs))

    includes_mol_df = includes_mol_df.drop(excludes_react_idxs)
    excludes_mol_df = excludes_mol_df.drop(includes_mol_idxs)

    return excludes_mol_df, includes_mol_df

def canonicalize_df(df):
    df['base_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in df['base_smiles']]
    df['ligand_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in df['ligand_smiles']]
    df['aryl_halide_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in df['aryl_halide_smiles']]
    df['additive_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in df['additive_smiles']]
    return df


def main():

    path = 'doyle_buchwald_data.csv'
    df = pd.read_csv(path)
    df = remove_nans(df)
    df = canonicalize_df(df)

    ranked_halides, ranked_ligands, ranked_bases, ranked_additives = find_mol_ranking(df)

    train_val_df, test_df = split_on_molecules(df, ranked_halides[0:4], 'halide')
    print(ranked_halides[0:4])
    print(test_df['aryl_halide_smiles'])
    print()
    print(train_val_df['aryl_halide_smiles'])

    train_val_df, test_df = split_on_molecule(df, ranked_ligands[1], 'ligand')
    print(ranked_ligands[1])
    print(test_df['ligand_smiles'])
    print()
    print(train_val_df['ligand_smiles'])

    train_val_df, test_df = split_on_molecule(df, ranked_bases[0], 'base')
    print(ranked_bases[0])
    print(test_df['base_smiles'])
    print()
    print(train_val_df['base_smiles'])

    train_val_df, test_df = split_on_molecules(df, ranked_additives[0:6], 'additive')
    print(ranked_additives[0:6])
    print(test_df['additive_smiles'])
    print()
    print(train_val_df['additive_smiles'])


if __name__ == '__main__':
    main()