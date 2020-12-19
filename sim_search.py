import time 
import selfies
import rdkit
import random
import numpy as np
import random
from rdkit import Chem
from selfies import encoder, decoder
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import Mol
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def randomize_smiles(mol):
    """
    Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.

    """
    if not mol:
        return None

    Chem.Kekulize(mol)
    
    return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True)



def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    

def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


class _FingerprintCalculator:
    """
    Calculate the fingerprint while avoiding a series of if-else.
    See recipe 8.21 of the book "Python Cookbook".

    To support a new type of fingerprint, just add a function "get_fpname(self, mol)".
    """

    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)

    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)


def get_fingerprint(mol: Mol, fp_type: str):
    return _FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)

def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 50% propbabily, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
    
    Parameters:
    selfie            (string)  : SELFIE string to be mutated 
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    
    while not valid:
        fail_counter += 1
                
        alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters 

        choice_ls = [1, 2, 3] # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]
        
        # Insert a character in a Random Location
        if random_choice == 1: 
            random_index = np.random.randint(len(chars_selfie)+1)
            random_character = np.random.choice(alphabet, size=1)[0]
            
            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character 
        elif random_choice == 2:                         
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
                
        # Delete a random character
        elif random_choice == 3: 
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]
                
        else: 
            raise Exception('Invalid Operation trying to be performed')

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)
        
        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()
    
    return (selfie_mutated, smiles_canon)

def get_mutated_SELFIES(selfies_ls, num_mutations): 
    for _ in range(num_mutations): 
        selfie_ls_mut_ls = []
        for str_ in selfies_ls: 
            
            str_chars = get_selfie_chars(str_)
            max_molecules_len = len(str_chars) + num_mutations
            
            selfie_mutated, _ = mutate_selfie(str_, max_molecules_len)
            selfie_ls_mut_ls.append(selfie_mutated)
        
        selfies_ls = selfie_ls_mut_ls.copy()
    return selfies_ls


def get_fp_scores(smiles_back, target_smi, fp_type): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    # fp_target = get_ECFP4(target)
    fp_target = get_fingerprint(target, fp_type)
    
    # init_  = Chem.MolFromSmiles('CCCCC')
    # target = Chem.MolFromSmiles('CCCCC')
    # init_fp = get_fingerprint(init_, 'AP')
    # target_fp = get_fingerprint(target, 'AP')
    
    # from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
    # score  = TanimotoSimilarity(init_fp, target_fp)

    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        # fp_mol = get_ECFP4(mol)
        fp_mol = get_fingerprint(mol, fp_type)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores
    
total_time = time.time()
num_random_samples = 50000 

repr_type          = 1 # 1=SELFIES; 2=Smiles; 3=DeepSmiles
num_mutation_ls    = [1, 2, 3, 4, 5]
# fp_type            = 'AP'
# fp_type            = 'FCFP4'
fp_type            = 'ECFP4'

# smi = 'C1CC(=O)NC2=C1C=CC(=C2)OCCCCN3CCN(CC3)C4=C(C(=CC=C4)Cl)Cl'   # Aripiprazole
# smi = 'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O'                            # Albuterol
# smi = 'CC12CCC3C(C1CCC2(C#C)O)CCC4=C3C=CC(=C4)OC'                   # Mestranol
smi = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'    # Celecoxib
# smi = 'C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O'              # Ciprofloxacin
# smi = 'C'

mol = Chem.MolFromSmiles(smi)
if mol == None: 
    raise Exception('Invalid starting structure encountered')
    
start_time = time.time()
randomized_smile_orderings  = [randomize_smiles(mol) for _ in range(num_random_samples)]

# Convert all the molecules to SELFIES
selfies_ls = [encoder(x) for x in randomized_smile_orderings]
print('Randomized molecules (in SELFIES) time: ', time.time()-start_time)


all_smiles_collect = []
all_smiles_collect_broken = []

start_time = time.time()
for num_mutations in num_mutation_ls: 
    # Mutate the SELFIE string: 
    selfies_mut = get_mutated_SELFIES(selfies_ls.copy(), num_mutations=num_mutations)
    
    # Convert back to SMILES: 
    smiles_back = [decoder(x) for x in selfies_mut]
    all_smiles_collect = all_smiles_collect + smiles_back
    all_smiles_collect_broken.append(smiles_back)
    
    # Get fingerprint scores for all of them: 
    # smiles_back_scores = get_fp_scores(smiles_back, target_smi=smi, fp_type=fp_type)
    
    # print('Num mutations: ', num_mutations)
    # print('    # New smile orderings = ', len(randomized_smile_orderings))
    # print('    Delta > 0.8: ', (len([x for x in smiles_back_scores if x > 0.8]) / len(randomized_smile_orderings)) * 100)
    # print('    Delta > 0.6: ', (len([x for x in smiles_back_scores if x > 0.6]) / len(randomized_smile_orderings)) * 100)
    # print('    Delta > 0.4: ', (len([x for x in smiles_back_scores if x > 0.4]) / len(randomized_smile_orderings)) * 100) 
print('Mutation obtainment time (back to smiles): ', time.time()-start_time)


# Work on:  all_smiles_collect
start_time = time.time()
canon_smi_ls = []
for item in all_smiles_collect: 
    mol, smi_canon, did_convert = sanitize_smiles(item)
    if mol == None or smi_canon == '' or did_convert == False: 
        raise Exception('Invalid smile string found')
    canon_smi_ls.append(smi_canon)
canon_smi_ls        = list(set(canon_smi_ls))
print('Unique mutated structure obtainment time: ', time.time()-start_time)

start_time = time.time()
canon_smi_ls_scores = get_fp_scores(canon_smi_ls, target_smi=smi, fp_type=fp_type)
print('Fingerprint calculation time: ', time.time()-start_time)
print('Total time: ', time.time()-total_time)

# THE TOP 100 MOLECULES ################################
a          = np.argsort(canon_smi_ls_scores)[::-1][:100]
scores_top = [canon_smi_ls_scores[i] for i in a ]
smi_top    = [canon_smi_ls[i] for i in a ]

thrh_score = []
for item in scores_top: 
    if item >= 0.75: 
        thrh_score.append(1.0)
    else: 
        thrh_score.append((4/3)*item)
# raise Exception('TESTING :) ')
score_1 = thrh_score[0]
score_10 = np.sum(thrh_score[0:10]) / 10
score_100 = np.sum(thrh_score[0:100]) / 100
print()
print('Total score: ', (score_1+score_10+score_100)/3)
# THE TOP 100 MOLECULES ################################

# print('    Delta > 0.8: ', len([x for x in canon_smi_ls_scores if x > 0.8]) )
A, B, C = [x for x in canon_smi_ls_scores if x > 0.75], [x for x in canon_smi_ls_scores if x > 0.6], [x for x in canon_smi_ls_scores if x > 0.4]

print('    Delta > 0.75: ', len(A) )
print('    Delta > 0.6: ', len(B))
print('    Delta > 0.4: ', len(C))

indices_thresh_8 = [i for i,x in enumerate(canon_smi_ls_scores) if x > 0.8]
mols_8 = [canon_smi_ls[idx] for idx in indices_thresh_8]


indices_thresh_6 = [i for i,x in enumerate(canon_smi_ls_scores) if x > 0.6 and x < 0.8]
mols_6 = [canon_smi_ls[idx] for idx in indices_thresh_6]

indices_thresh_4 = [i for i,x in enumerate(canon_smi_ls_scores) if x > 0.4 and x < 0.6]
mols_4 = [canon_smi_ls[idx] for idx in indices_thresh_4]

print('    Delta > 0.75: ', (len(A)/len(canon_smi_ls_scores))*100 )
print('    Delta > 0.6: ', (len(B)/len(canon_smi_ls_scores))*100)
print('    Delta > 0.4: ', (len(C)/len(canon_smi_ls_scores))*100)


# Period -> newline
# Header-> smiles

with open('./right.txt', 'w') as f: 
    f.writelines(['smiles\n', '\n'.join(x for x in mols_4)])








