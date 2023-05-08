import pickle
import py3Dmol
import pandas as pd
import numpy as np
import streamlit as st
from stmol import showmol
# importing from rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


def makeblock(smi):
    if valid_smiles(smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
            mblock = Chem.MolToMolBlock(mol)
        except ValueError:
            return None
        return mblock

def render_mol(xyz,style):
    xyzview = py3Dmol.view(width=650,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({style.lower():{}})
    xyzview.setBackgroundColor('#0E1117')  
    xyzview.zoomTo()
    showmol(xyzview,height=400,width=650)
    
def render_prot(xyz,style,spin, width=650, height=400):
    
    view = py3Dmol.view(query=f'pdb:{xyz}', width=width, height=height)
    view.setStyle({style.lower():{'color':'spectrum'}})
    view.setBackgroundColor('#0E1117')
    view.rotate(270, 'x')
    view.spin((0, 0, 1, 0))
    view.zoomTo()
    showmol(view, width=width, height=height)
    
@st.cache
def computeFP(smiles, labels=None):
    moldata = [Chem.MolFromSmiles(mol) for mol in smiles]
    fpdata=[]
    for i, mol in enumerate(moldata):
        if mol:
            ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
            fpdata += [ecfp6]
    fp_df = pd.DataFrame(data=fpdata, index=smiles)
    if labels is not None: fp_df['labels'] =labels
    return fp_df


def valid_smiles(smi):
    if Chem.MolFromSmiles(smi) is not None:
        return True
    else:
        return False


@st.cache
def predict_gpcr(smi):
    with open('gpcr_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
        mol = Chem.MolFromSmiles(smi)
        ecfp6 = [[int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]]
        prediction=model.predict(ecfp6)
        return prediction
        
        

