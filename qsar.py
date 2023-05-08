import pickle
import numpy as np
import pandas as pd
import custom_funcs as ctk
import statistics as sts
import matplotlib.pyplot as plt
from collections import Counter
from chembl_webresource_client.new_client import new_client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.metrics import geometric_mean_score
from skopt import BayesSearchCV
from rdkit import Chem
from rdkit.Chem import AllChem
from skopt.space import Integer, Categorical
from imblearn.under_sampling import NearMiss
from sklearn.manifold import TSNE

ensemble = {
    'rf': RandomForestClassifier,
    'svm': SVC,
    'mlp': MLPClassifier
}
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


    

def deploy_model(key,smiles):
    data = []
    with open(f'models/{key}.pkl', 'rb') as f:
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
            data += [ecfp6]
        model = pickle.load(f)
        res = model.predict(data)
    return list(res)
        
def deploy_ensemble(smiles):
    predictions = {}
    for key in ensemble:
        pred = deploy_model(key, smiles)
        predictions[key] = pred
    if len(predictions[key]) > 1:
        return predictions
    else:
        gpcrs = {}
        for key in predictions:
            code = predictions[key][0]
            gpcrs[key] = gpcr_encoded[code]
        return gpcrs

def prediction_confidences(smiles, predictions):
    confidences = []
    rf_preds = predictions['rf']
    svm_preds = predictions['svm']
    mlp_preds = predictions['mlp']
    for x, y, z in zip(rf_preds,svm_preds,mlp_preds):
        if x == y and y == z:
            confidence = 'High'
        elif x != y and y != z and z != x:
            confidence = 'None'
        else:
            confidence = 'Low'
        confidences.append(confidence)
    predictions['confidence'] = confidences
    predictions['smiles'] = smiles
    res = pd.DataFrame(predictions)
    res = res.iloc[:, ::-1]
    return res
        
    





        
    
