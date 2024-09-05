from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import sys
from rdkit import Chem
#from rdkit.Chem import Draw, PandasTools
#import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdchem

#Edge Feature
@dataclass
class BondInfo:
    is_conjugated: List[bool] = field(default_factory=lambda: [True]) #가변 데이터 타입의 필드에 기본값을 지정(list, set, dict)
    bond_type: dict = field(default_factory=lambda: {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0, 0],
        Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1, 0],
        Chem.rdchem.BondType.UNSPECIFIED: [0, 0, 0, 0, 1]
    })
    
@dataclass
class EdgeFeat:
    smiles : str
    mol : Chem.Mol = field(init=False) #나중에 따로 입력을 받아야함. __post_init__사용
    
    def __post_init__(self):
        self.mol = Chem.MolFromSmiles(self.smiles)
    
    def BndIdx(self) -> List[Tuple[int,int]]:
        return [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in self.mol.GetBonds()]
    
    def NumRot(self) -> List[int]:
        rot_idx= Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        rot_num = self.mol.GetSubstructMatches(rot_idx)
        rot_list=[0]*len(self.BndIdx())
        for i in range(len(self.BndIdx())):
            if self.BndIdx()[i] in rot_num:
                rot_list[i]=1
        return rot_list 
     
    def IsConjugated(self, bond_info: BondInfo) -> List[bool]:
        return [b.GetIsConjugated() == bond_info.is_conjugated[0] for b in self.mol.GetBonds()]
    
    def BndInfo(self, bond_type: BondInfo) -> List[List[int]]:
        return [bond_type.bond_type[b.GetBondType()] for b in self.mol.GetBonds()]

#Node Feature  
@dataclass
class NodeOneHot:
    OneHotHybrid: dict = field(default_factory= lambda:{
        rdchem.HybridizationType.S : [1,0,0,0,0],
        rdchem.HybridizationType.SP: [0,1,0,0,0],
        rdchem.HybridizationType.SP2: [0,0,1,0,0],
        rdchem.HybridizationType.SP3: [0,0,0,1,0],
        rdchem.HybridizationType.UNSPECIFIED: [0,0,0,0,1],
        rdchem.HybridizationType.OTHER: [0,0,0,0,1],
        rdchem.HybridizationType.SP2D: [0,0,0,0,1],
        rdchem.HybridizationType.SP3D: [0,0,0,0,1],
        rdchem.HybridizationType.SP3D2: [0,0,0,0,1]
    })
    
    func_listing: dict = field(default_factory= lambda: {'fr_Al_OH':0, #아마이드가 합성이 쉽다.
               'fr_Ar_OH':1,
               'fr_ester':2,
               'fr_ketone':3,
               'fr_ether':4,
               'fr_amide':5
               ,'fr_C_S':6
               ,'fr_sulfone':7
               ,'fr_sulfone2':8
               ,'fr_urea':9
               ,'fr_lactone':10
               ,'fr_phos_acid':11
               ,'fr_alkyl_carbamate':12
               ,'unknown':13})
    
    is_aromatic: List[bool] = field(default_factory= lambda: [True])
    
    is_ring: List[bool] = field(default_factory= lambda: [True])
    

@dataclass
class NodeFeat:
    # Aromatic, NumTermCH3, NumRing, GetHybrid, FuncG
    smiles: str
    mol: Chem.Mol = field(init=False)

    def __post_init__(self):
        self.mol = Chem.MolFromSmiles(self.smiles)
    
    def Aromatic(self, aroma: NodeOneHot) -> List[bool]:
        return [ring.GetIsAromatic()==aroma.is_aromatic[0] for ring in self.mol.GetAtoms()]

    def NumCH3(self) -> List[bool]:
        terminalCH3=Chem.MolFromSmarts("[CX4H3]")
        calCH3=[False]*self.mol.GetNumAtoms()
        match = self.mol.GetSubstructMatches(terminalCH3)
        for i in match:
            calCH3[i[0]]=True
        return calCH3
    
    def Ring(self, ring: NodeOneHot) -> List[bool]:
        return [atom.IsInRing()==ring.is_ring[0] for atom in self.mol.GetAtoms()]
    
    def Hybrid(self, hiv: NodeOneHot) -> List[List[int]]:
        return [hiv.OneHotHybrid[atom.GetHybridization()] for atom in self.mol.GetAtoms()]
    
    def OneHotFuncG(self, func_dict: dict,funclist:NodeOneHot) -> List[List[int]]:
        func_list=func_dict
        one_hot=[[0]*len(funclist.func_listing) for _ in range(self.mol.GetNumAtoms())]
        for func, indices in func_list.items():
            func_idx = funclist.func_listing[func]
            for idx in indices:
                one_hot[idx][func_idx]=1
        return one_hot
    '''
    def get_substructure_atoms(self, smarts_pattern:NodeOneHot) -> List[int]:
        pattern = Chem.MolFromSmarts(smarts_pattern)
        matches = self.mol.GetSubstructMatches(pattern)
        atom_indices=[idx for match in matches for idx in match]
        return atom_indices

    def FuncG(self, func: NodeOneHot) -> dict:
        if self.mol is not None:
            LinearAmide_atoms = self.get_substructure_atoms(func.func_type['LinearAmide'])
            LinearEster_atoms = self.get_substructure_atoms(func.func_type['LinearEster'])
            LinearThioamide_atoms = self.get_substructure_atoms(func.func_type['LinearThioamide'])
            LinearThioester_atoms = self.get_substructure_atoms(func.func_type['LinearThioester'])
            CyclicAmide_atoms = self.get_substructure_atoms(func.func_type['CyclicAmide'])
            CyclicEster_atoms = self.get_substructure_atoms(func.func_type['CyclicEster'])
            CyclicThioamide_atoms = self.get_substructure_atoms(func.func_type['CyclicThioamide'])
            CyclicThioester_atoms = self.get_substructure_atoms(func.func_type['CyclicThioester'])
            Alcohol_atoms = self.get_substructure_atoms(func.func_type['alcohol'])
            Phosphonate_atoms = self.get_substructure_atoms(func.func_type['phosphonate'])
            Carbamate_atoms = self.get_substructure_atoms(func.func_type['carbamate'])
            Urea_atoms = self.get_substructure_atoms(func.func_type['urea']) 
            Sulfonate_atoms = self.get_substructure_atoms( func.func_type['sulfonate'])
            Ketone_atoms = self.get_substructure_atoms( func.func_type['ketone'])
            Ether_atoms = self.get_substructure_atoms( func.func_type['ether'])

            total_atoms = [idx for idx in range(self.mol.GetNumAtoms())]
        
            known_atoms = set(LinearAmide_atoms + LinearEster_atoms + LinearThioamide_atoms + LinearThioester_atoms+CyclicAmide_atoms+CyclicEster_atoms+
                              CyclicThioamide_atoms + CyclicThioester_atoms + Alcohol_atoms + Phosphonate_atoms + Carbamate_atoms +
                              Urea_atoms + Sulfonate_atoms + Ketone_atoms + Ether_atoms)
            unknown_atoms = [i for i in (total_atoms) if i not in known_atoms]
            
            return {"LinearAmide": LinearAmide_atoms,
                     "LinearEster": LinearEster_atoms,
                     "LinearThioamide": LinearThioamide_atoms,
                     "LinearThioester": LinearThioester_atoms,
                     "CyclicAmide":CyclicAmide_atoms,
                     "CyclicEster":CyclicEster_atoms,
                     "CyclicThioamide":CyclicThioamide_atoms,
                     "CyclicThioester":CyclicThioester_atoms,
                     "Alcohol":Alcohol_atoms,
                     "Phosphonate":Phosphonate_atoms,
                     "Carbamate":Carbamate_atoms,
                     "Urea":Urea_atoms,
                     "Sulfonate":Sulfonate_atoms,
                     "Ketone":Ketone_atoms,
                     "Ether":Ether_atoms,
                     "Unknown": unknown_atoms}
    
    def _CountMatches(self,patt,unique=True):
      return self.mol.GetSubstructMatches(patt,uniquify=unique)
    
    
    def _LoadPatterns(self,fileName=None):
        fns=[]
        check_list: list =['fr_lactone','fr_amide','fr_ester','fr_C=S','fr_phos_acid','fr_alkyl_carbamate','fr_urea','fr_sulfone','fr_sulfone2','fr_ketone','fr_ether','fr_Al_OH','fr_Ar_OH']
        defaultPatternFileName: str =('/home/kathy531/Caesar-lig/code/rdkit/FragmentsDescriptors.csv')
        if fileName is None:
            fileName = defaultPatternFileName
        try:
            #print(fileName)
            inF = open(fileName,'r')
        except IOError:
            raise IOError
        else:
            for line in inF.readlines():
                #print(line)
                if len(line) and line[0] != '#':
                    splitL = line.split('\t')
                    if len(splitL)>=3:
                        name = splitL[0]
                        if name not in check_list: continue
                        descr = splitL[1]
                        sma = splitL[2]
                        descr=descr.replace('"','')
                        ok=1
                        try:
                            patt = Chem.MolFromSmarts(sma) #smart -> mol
                        except:
                            ok=0
                        else:
                            if not patt or patt.GetNumAtoms()==0: ok=0
                        if not ok: raise ImportError#'Smarts %s could not be parsed'%(repr(sma))
                        fn = lambda mol,countUnique=True,pattern=patt:self._CountMatches(mol,pattern,unique=countUnique)
                        fn.__doc__ = descr
                        name = name.replace('=','_')
                        name = name.replace('-','_')
                        fns.append((name,fn))
        return fns
    
    def FuncG_rev(fns,mol):
        func_dict={}
        known_atoms=[]
        if fns is not None:
            for name, fn in fns:
                if name not in func_dict:
                    func_dict[name]=[]
                #print(fn)
                for i in fn(mol):
                    for j in i:
                        func_dict[name].append(j)
                        known_atoms.append(j)
        total_atoms = [idx for idx in range(mol.GetNumAtoms())]
        unknown_atoms = [idx for idx in total_atoms if idx not in known_atoms]
        func_dict['unknown'] = unknown_atoms 
        return func_dict'''
    

