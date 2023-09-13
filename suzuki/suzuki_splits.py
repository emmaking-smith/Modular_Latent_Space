'''
Split the dataset randomly, based on ligand, or based on reactants (boronic acid / aryl bromide).
'''


import re
from rdkit import Chem
from collections import Counter
from itertools import chain

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def load_parsed_data(path, start_index=0):
    '''
    Returns a dictonary whose keys are the
    rxn ids and whose values are dictionaries containing
    that rxn's information.
    '''
    dict = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            dict[i + start_index] = eval(line)
    return dict

def find_reactants(rxn):
    '''
    Removes that atom numbering and orders reactants such that
    the nucleophile (boronic acid) is entry 0, and electrophile (aryl halide)
    is entry 1.
    '''
    reactants = [rxn[0][x] for x in range(len(rxn[0])) if type(rxn[0][x]) == str]
    nucleophile = 0
    electrophile = 0
    for reactant in reactants:
        if 'B(' in reactant:
            nucleophile = reactant
        else:
            electrophile = reactant
    assert nucleophile != 0 and electrophile != 0
    ordered_canon_reactants = [remove_atom_mapped_numbering(nucleophile), remove_atom_mapped_numbering(electrophile)]
    return ordered_canon_reactants

def remove_atom_mapped_numbering(smiles):
    smiles = re.sub(':[0-9]+', '', smiles)
    canonical_smiles = canonicalize_smiles(smiles)
    return canonical_smiles

def split_on_reactant(rxn_dict, canon_reactant_smiles):
    '''
    Given a reactant to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''
    excludes_reactant_dict = rxn_dict.copy()
    includes_reactant_dict = rxn_dict.copy()
    includes_reactant_idxs = []
    for key in list(rxn_dict.keys()):
        reactants = find_reactants(rxn_dict[key]['rx'])
        if canon_reactant_smiles in reactants:
            includes_reactant_idxs.append(key)
    excludes_react_idxs = list(set(list(rxn_dict.keys())) - set(includes_reactant_idxs))
    for key in includes_reactant_idxs:
        del excludes_reactant_dict[key]
    for key in excludes_react_idxs:
        del includes_reactant_dict[key]

    return excludes_reactant_dict, includes_reactant_dict

def split_on_multiple_electrophiles(rxn_dict, canon_electrophile_smiles_list):
    '''
    Splits into training / testing / validation sets for multiple electrophile smiles.
    '''
    excludes_reactant_dict = rxn_dict.copy()
    includes_reactant_dict = rxn_dict.copy()
    includes_reactant_idxs = []
    for key in list(rxn_dict.keys()):
        reactants = find_reactants(rxn_dict[key]['rx'])
        if len(list(set(canon_electrophile_smiles_list) & set(reactants))) > 0:
            includes_reactant_idxs.append(key)
    excludes_react_idxs = list(set(list(rxn_dict.keys())) - set(includes_reactant_idxs))
    for key in includes_reactant_idxs:
        del excludes_reactant_dict[key]
    for key in excludes_react_idxs:
        del includes_reactant_dict[key]

    return excludes_reactant_dict, includes_reactant_dict

def split_on_cat_lig(rxn_dict, canon_cat_lig_smiles):
    '''
    Given a catalyst or ligand to exclude for test set, indentify
    all rxns that utilize that reactant and separate them out.
    '''
    excludes_reactant_dict = rxn_dict.copy()
    includes_reactant_dict = rxn_dict.copy()
    includes_reactant_idxs = []
    for key in list(rxn_dict.keys()):
        catalyst, ligand = find_catalysts_ligands(rxn_dict[key])
        catalyst = reformat_ligs_and_cats([catalyst])
        catalyst = canonicalize_smiles(catalyst[0])

        ligand = clean_ligands([ligand])
        ligand = reformat_ligs_and_cats(ligand)
        ligand = canonicalize_smiles(ligand[0])

        cat_lig = [catalyst, ligand]

        if canon_cat_lig_smiles in cat_lig:
            includes_reactant_idxs.append(key)
    excludes_react_idxs = list(set(list(rxn_dict.keys())) - set(includes_reactant_idxs))
    for key in includes_reactant_idxs:
        del excludes_reactant_dict[key]
    for key in excludes_react_idxs:
        del includes_reactant_dict[key]

    return excludes_reactant_dict, includes_reactant_dict

def ranking_nucleophiles_and_electrophiles(rxn_dict):
    '''
    Finding most common electrophiles and nucleophiles used in rxns.
    '''
    nucleophiles = [find_reactants(rxn_dict[x]['rx'])[0] for x in list(rxn_dict.keys())]
    electrophiles = [find_reactants(rxn_dict[x]['rx'])[1] for x in list(rxn_dict.keys())]
    most_common_nucleophiles = Counter(nucleophiles).most_common()[0:2] # OB(O)c1ccccc1 most common nucleophile
    most_common_electrophiles = Counter(electrophiles).most_common()[0:60] # Brc1ccccn1 most common electrophile
    return [x[0] for x in most_common_nucleophiles], [x[0] for x in most_common_electrophiles]

def ranking_catalysts_and_ligands(rxn_dict):
    '''
    Finding most common catalysts. Note that catalysts do NOT have atom mapping to get rid of.
    '''
    # Catalysts
    catalysts = [find_catalysts_ligands(rxn_dict[x])[0] for x in list(rxn_dict.keys())]
    catalysts = reformat_ligs_and_cats(catalysts)
    catalysts = [canonicalize_smiles(x) for x in catalysts]
    most_common_catalysts = Counter(catalysts).most_common()[1:3]

    # Ligands
    ligands = [find_catalysts_ligands(rxn_dict[x])[1] for x in list(rxn_dict.keys())]
    ligands = clean_ligands(ligands)
    ligands = reformat_ligs_and_cats(ligands)
    ligands = [canonicalize_smiles(x) for x in ligands]
    most_common_ligands = Counter(ligands).most_common()[1:3] # most common ligand is no ligand at all.

    return [x[0] for x in most_common_catalysts], [x[0] for x in most_common_ligands]


def find_catalysts_ligands(rxn):
    '''
    Finding the catalysts and ligands used in each reaction.
    Note that some rxns may not use ligands (e.g. just the catalyst / a precatalyst was used).
    '''
    catalyst = rxn['catalysts']
    ligand = list(rxn['ligands'])
    return catalyst, ligand

def detuple(set_of_tuple):
    return list(chain.from_iterable(set_of_tuple))

def clean_ligands(ligand_list):
    cleaned_ligands = []
    for ligs in ligand_list:
        if len(ligs) == 0:
            cleaned_ligands.append(ligs)
        elif len(ligs) == 1:
            if type(list(ligs)[0]) == tuple:
                cleaned_ligands.append(detuple(ligs))
            else:
                cleaned_ligands.append(ligs)
        else:
            mixed_tuple_ligands = []
            for l in ligs:
                if type(l) != str:
                    mixed_tuple_ligands = mixed_tuple_ligands + list(l)
                else:
                    mixed_tuple_ligands.append(l)

            cleaned_ligands.append(mixed_tuple_ligands)
    return cleaned_ligands

def reformat_ligs_and_cats(cleaned_ligs_or_cats):
    reformatted = []
    for smiles_list in cleaned_ligs_or_cats:
        reformat = '.'.join(smiles_list)
        reformatted.append(reformat)
    return reformatted

def main():

    path = 'suzuki_from_arom_USPTO_parsed_het.txt'
    dict = load_parsed_data(path)
    # most_common_nucleophiles, most_common_electrophiles = ranking_nucleophiles_and_electrophiles(dict)
    # print(most_common_nucleophiles)
    # train_val_dict, test_dict = split_on_reactant(dict, most_common_nucleophiles[0])
    # train_dict, val_dict = split_on_reactant(train_val_dict, most_common_nucleophiles[1])
    # print(len(train_dict), len(val_dict), len(test_dict))
    #
    # most_common_nucleophiles, most_common_electrophiles = ranking_nucleophiles_and_electrophiles(dict)
    # print(most_common_electrophiles)
    # train_val_dict, test_dict = split_on_multiple_electrophiles(dict, most_common_electrophiles[0:35])
    # train_dict, val_dict = split_on_multiple_electrophiles(train_val_dict, most_common_electrophiles[35:])
    # print(len(train_dict), len(val_dict), len(test_dict))

    most_common_catalysts, most_common_ligands = ranking_catalysts_and_ligands(dict)
    print(most_common_catalysts)
    train_val_dict, test_dict = split_on_cat_lig(dict, most_common_catalysts[0])
    train_dict, val_dict = split_on_cat_lig(train_val_dict, most_common_catalysts[1])
    print(len(train_dict), len(val_dict), len(test_dict))

    # train_val_dict, test_dict = split_on_cat_lig(dict, most_common_ligands[0])
    # print(most_common_ligands)
    # train_dict, val_dict = split_on_cat_lig(train_val_dict, most_common_ligands[1])
    # print(len(train_dict), len(val_dict), len(test_dict))






if __name__ == '__main__':
    main()