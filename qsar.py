import pickle
import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


params = {
    'rf': {
        'n_estimators': 1000,
        'min_samples_leaf': 3,
        'criterion': 'entropy'
    },
    'svm': {
            'kernel':'rbf', 
            'C': 100,
    },
    'mlp': {
            'alpha': 0.1,
            'hidden_layer_sizes': 1000,
            'max_iter': 1000
    }
}

gpcr_encoded = {
    1: '5-hydroxytryptamine receptor',
    2: 'Adenosine receptor',
    3:  'Alpha-adrenergic receptor',
    4: 'Angiotensin receptor',
    7: 'B-Bradykinin receptor',
    8: 'Beta adrenergic receptor',
    10: 'C-C Chemokine receptor',
    11: 'C-X-C Chemokine receptor',
    16: 'Cannabinoid receptor',
    18: 'Cholecystokinin receptor',
    19: 'Corticotropin-releasing factor receptor',
    21: 'Dopamine receptor',
    22: 'Endothelin receptor',
    27: 'G-protein coupled receptor',
    36: 'Gonadotropin-releasing hormone receptor',
    37: 'Growth hormone secretagogue receptor',
    38: 'Histamine receptor',
    44: 'Melanin-concentrating hormone receptor',
    45: 'Melanocortin receptor',
    48: 'Metabotropic glutamate receptor',
    50: 'Muscarinic acetylcholine receptor',
    54: 'Neuropeptide Y receptor',
    58: 'Opioid receptor',
    59: 'Orexin receptor',
    62: 'P2Y purinoceptor ',
    64: 'Platelet-activating factor receptor',
    69: 'Prostaglandin receptor',
    70: 'Proteinase-activated receptor',
    75: 'Sphingosine 1-phosphate receptor',
    77: 'Substance-P receptor',
    86: 'Vasopressin receptor'
}


@st.cache
def load_model():
    return pickle.load(open(f'models/svm.pkl', 'rb'))

def deploy_model(smiles):
    data = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
        data += [ecfp6]
    model = load_model()
    res = model.predict(data)
    return list(res)
        
        
    





        
    
