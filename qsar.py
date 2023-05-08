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

gpcr_pdbs = {                                                                               
    1: '7E2X',
    2: '5N2S',
    3: '6KUW'   
}

def main():
    df = pd.read_csv('ecfp6_data.csv')
    X = df.drop('labels', axis=1)
    y = df['labels']
    nm = NearMiss(sampling_strategy=under_sampling, n_jobs=-1)
    X_res, y_res = nm.fit_resample(X,y)
    train_ensemble(X_res,y_res)
    
def train_ensemble(X, y):
    for key in list(ensemble.keys()):
        estimator = ensemble[key]
        hyperparameters = params[key]
        model = train_model(X,y,estimator, hyperparameters)
        with open(f'{key}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
def train_model(X, y, estimator, params):
    STATE = 999
    model = estimator(**params, random_state=STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, shuffle=True, random_state=STATE)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test,predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    mcc = matthews_corrcoef(y_test,predictions)
    print('acc ', acc)
    print('f1 ', f1)
    print('mcc ', mcc)
    print(set(y))
    return model

def chembl_target_activity(tar,activity):
    client = new_client.activity
    activities = client.filter(target_chembl_id=tar, standard_type=activity)
    if activities == []:
        return None
    df = pd.DataFrame.from_dict(activities)
    df = df[df.standard_value.notna()]
    mol_ids = [cid for cid in df.molecule_chembl_id]
    smiles = [mol for mol in df.canonical_smiles]
    std_vals = [float(val) for val in df.standard_value]
    labels =[]
    for val in std_vals:
        if val > sts.median(std_vals):
            labels.append(0)
        else:
            labels.append(1)
    tuples = list(zip(mol_ids,smiles,std_vals, labels))
    df = pd.DataFrame(tuples, columns=['chembl_id', 'smiles','value', 'labels'])
    df.drop_duplicates(subset='chembl_id')
    return df


def under_sampling(y):
    sample_dict = {}
    for i in set(y):
        if np.count_nonzero(y == i) > 1000:
            sample_dict[i] = 1000
        elif np.count_nonzero(y < 100):
            sample_dict[i] = 0
    return sample_dict

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
        
        
def generate_tsne():
    df = pd.read_csv('ecfp6_data.csv')
    X = df.drop('labels', axis=1)
    y = df['labels']
    nm = NearMiss(sampling_strategy=under_sampling, n_jobs=-1)
    X_res, y_res = nm.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,
                                                        train_size=0.70,
                                                        test_size=0.30,
                                                        random_state=999,
                                                        stratify=y_res)

    X_train = [list(i) for i in X_train]
    X_test = [list(i) for i in X_test]
    X_col = [1 for i in range(len(X_train))] + [0 for i in range(len(X_test))]

    tSNE_data = TSNE(n_components=3, n_jobs=-1, verbose=1).fit_transform(X_res)
    tSNE_x, tSNE_y, tSNE_z = list(zip(*tSNE_data))
    tSNE_fig = plt.figure(figsize=(8,8))
    ax1 = tSNE_fig.add_axes([0,0,1,1],projection='3d')
    ax1.grid(color='white')
    ax1.set_xlabel('tSNE-1')
    ax1.set_ylabel('tSNE-2')
    ax1.set_zlabel('tSNE-3')
    ax1.scatter(tSNE_x, tSNE_y, tSNE_z, s=60, c=y_res, cmap='BuPu', linewidth=1, edgecolor='black')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.zaxis.label.set_color('white')
    ax1.title.set_color('white')
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.tick_params(axis='z', colors='white')
    ax1.xaxis._axinfo["grid"]['color'] =  'white'
    ax1.yaxis._axinfo["grid"]['color'] =  'white'
    ax1.zaxis._axinfo["grid"]['color'] =  'white'

    tSNE_fig.patch.set_alpha(0)
    ax1.patch.set_alpha(0)
    with open('tsne.pkl', 'wb') as f:
        pickle.dump(tSNE_fig, f)
    tSNE_fig.savefig('tsne.pdf')
    plt.show()


if __name__ == '__main__':
    rf_acc= 0.9467741935483871
    rf_f1 =0.9468783432101854
    rf_mcc =0.9450155358669787

    svm_acc = 0.9706451612903226
    svm_f1 = 0.9707199225832382
    svm_mcc = 0.9696760854299427

    mlp_acc = 0.9567741935483871
    mlp_f1 = 0.9568833547336512
    mlp_mcc = 0.9553597002552187
    
    metrics = ['acc', 'f1', 'mcc']
    acc_metrics = [rf_acc,svm_acc,mlp_acc]
    f1_metrics= [rf_f1,svm_f1,mlp_f1]
    mcc_metrics= [rf_mcc,svm_mcc,mlp_mcc]
    width = 0.15
    fig, ax = plt.subplots()
    ax.bar(np.arange(3), acc_metrics, width = width, label='Accuracy')
    ax.bar(np.arange(3) + width, f1_metrics, width= width, label='F1-score')
    ax.bar(np.arange(3) + 2*width, mcc_metrics, width= width, label='MCC')
    ax.set_xlabel('Model')
    ax.set_ylabel('Performance')
    ax.set_title('Model Metrics')
    ax.set_xticks(np.arange(3) + width/2, ['RF', 'SVM', 'MLP'])
    ax.legend()
    plt.show()
    

# with open('gpcr_classifier.pkl', 'rb') as f:
#     model = pickle.load(f)
#     df = pd.read_csv('ecfp6_data.csv')
#     X = df.drop('labels', axis=1)
#     y = df['labels']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y, shuffle=True, random_state=999)
#     predictions = model.predict(X_test)
#     acc = accuracy_score(y_test,predictions)
#     f1 = f1_score(y_test, predictions, average='micro')
#     mcc = matthews_corrcoef(y_test,predictions)
#     print('acc ', acc)
#     print('f1 ', f1)
#     print('mcc ', mcc)



        
    
