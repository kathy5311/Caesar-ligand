import numpy as np
import sys
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdchem
from feature_func_0722 import BondInfo, EdgeFeat, NodeOneHot, NodeFeat

ELEMS_TYPE = ['C','O','N','P','S','Br','I','F','Cl']

# Use smiles function(Using .sdf file)
def methyl(sdf_path, j):   
    # SDF path
    #1~30까지 자동화 시키려면 sdf_path에서 subset 숫자만 따 와서 dir_path에 지정
    dir_path = '/ml/crest-zinc/subset_%d/'%(j)
    '''
    sdf_path_list = sdf_path.split("_")
    sdf_path = sdf_path_list[2]+"_"+sdf_path_list[-1]
    '''
    total_path=dir_path+sdf_path+'.sdf'

    # Load SDF
    supplier = Chem.SDMolSupplier(total_path)
    
    # supplier to canonical SMILES
    for mol in supplier:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol, canonical=True)
    
            one_hot_node=NodeOneHot()
            node= NodeFeat(smiles)
            one_hot_edge=BondInfo()
            edge=EdgeFeat(smiles)
            
            #Node
            #Aromatic Ring
            Aromatic_atom = node.Aromatic(one_hot_node)
            #NumCH3
            calCH3 = node.NumCH3()
            #Ring
            Ring_check = node.Ring(one_hot_node)
            #Hybrid
            Hybrid_check = node.Hybrid(one_hot_node)
            #OneHotFuncG
            FuncG_check = node.OneHotFuncG(one_hot_node)
            
            #Edge
            #BndIdx
            Bond_idx = edge.BndIdx()
            #Conjugated
            Is_conju = edge.IsConjugated(one_hot_edge)
            #BndInfo
            Bond_info = edge.BndInfo(one_hot_edge)
            #Rotbond
            Num_rot = edge.NumRot()
    
    return Aromatic_atom, calCH3, Ring_check, Hybrid_check, FuncG_check, Bond_idx, Is_conju, Bond_info, Num_rot
    
def read_npz(npzfile, outf, j):
    dirpath='/home/kathy531/EntropyNet/ligand/EnergyNet/features/'
    filepath=dirpath+npzfile
    data= np.load(filepath, allow_pickle=True)
    tag_list=list(data.keys())
    
    features=[]
    #tag_list 수정해야해 -> subset별 tag 달아주기
    for i in range(len(tag_list)):
        #if tag_list[i].split("|")[-1] not in ['mol_1', 'mol_2']: continue
        tag_list[i]=f"subset{j}|{tag_list[i]}"
        print(tag_list[i].split("|")[-1])
        print("processing")
        Aromatic_atom, calCH3, Ring_check, Hybrid_check, FuncG_check, Bond_idx, Is_conju, Bond_info, Num_rot = methyl(tag_list[i].split("|")[-1],j)#[0]
        array=data[tag_list[i].split("|")[-1]].tolist()['elems']
        energy=data[tag_list[i].split("|")[-1]].tolist()['energy']
        xyz=data[tag_list[i].split("|")[-1]].tolist()['xyz']
        ent=data[tag_list[i].split("|")[-1]].tolist()['ent']
        numH=data[tag_list[i].split("|")[-1]].tolist()['numH']
        #new_bond=data[tag_list[i].split("|")[-1]].tolist()['new_bond']
        elem_type = np.array(ELEMS_TYPE)
        
        element_to_index = {element: index for index, element in enumerate(elem_type) if element!='H'}

        #indices = np.array([element_to_index[element] for element in array if element != "H"])
        indices = []
        for element in array:
            if element != "H":
                try:
                    indices.append(element_to_index[element])
                except KeyError:
                    print(f"KeyError: '{element}' is not found")
        indices = np.array(indices)
        one_hot_encoded = np.zeros((indices.size, elem_type.size), dtype=int)
        one_hot_encoded[np.arange(indices.size), indices] = 1
        
        features.append({'bond_idx': np.array(Bond_idx), 'bond_info': np.array(Bond_info), 'elems' : one_hot_encoded, 'energy' : energy, 'xyz' : xyz, 
                         'ent' : ent, 'numH': numH, 'aromatic': np.array(Aromatic_atom), 'numCH3' : np.array(calCH3), 'rotBond' : np.array(Num_rot), 
                         'ring': np.array(Ring_check), '1hotFuncG': np.array(FuncG_check), '1hotHybrid': np.array(Hybrid_check), 'conjugated' : np.array(Is_conju) })
        np.savez(outf, tags=tag_list, features=features)

def multiP(j):
    npzfile ='subset_%d_new_rev.prop.npz'%(j)
    read_npz(npzfile, '../new_npz0718/test_subset_rev0718_%d.npz'%(j),j)
    print(f"Done{j}")




if __name__ == "__main__":
    #자동화 하려면 subset의 숫자만 따와서 for문 돌리기
    with mp.Pool(processes=92) as pool:
        pool.map(multiP, range(1,31))
