# $Id$
#
# Copyright (C) 2002-2010 greg Landrum and Rational Discovery LLC
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
""" functions to match a bunch of fragment descriptors from a file

No user-servicable parts inside.  ;-)

"""
import os
import rdkit
from rdkit import Chem

check_list=['fr_lactone','fr_amide','fr_ester','fr_C=S','fr_phos_acid','fr_alkyl_carbamate','fr_urea','fr_sulfone','fr_sulfone2','fr_ketone','fr_ether','fr_Al_OH','fr_Ar_OH']

defaultPatternFileName =('FragmentsDescriptors.csv')

def _CountMatches(mol,patt,unique=True):
  return mol.GetSubstructMatches(patt,uniquify=unique)

fns = []
def _LoadPatterns(fileName=None):
    if fileName is None:
        fileName = defaultPatternFileName
    try:
        print(fileName)
        inF = open(fileName,'r')
    except IOError:
        print("error")
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
                    fn = lambda mol,countUnique=True,pattern=patt:_CountMatches(mol,pattern,unique=countUnique)
                    fn.__doc__ = descr
                    name = name.replace('=','_')
                    name = name.replace('-','_')
                    fns.append((name,fn))
    return fns
                    #print(fn)
mol=Chem.MolFromSmiles('C=1CCCCC1')
fns=_LoadPatterns(fileName='./FragmentsDescriptors.csv')
    

for name,fn in fns:
    print(fn(mol))
    print(name)
    #(exec('%s=fn'%(name))
    
#fn=None

