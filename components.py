import numpy as np
import pandas as pd
import streamlit as st
import custom_funcs as ctk
import qsar
from collections import Counter
from io import StringIO

def about():
    
    st.markdown('''
                ### Learn more about GPCRs (PDB ID: 7WIH)
                ''')
    with st.container():
        ctk.render_prot('7WIH','cartoon',True, height=300)
    st.markdown('''
G protein-coupled receptors (GPCRs) are an essential class of transmembrane proteins that play a crucial role in the regulation of a wide range of physiological processes. As a result, GPCRs have become a vital target for drug discovery, and many pharmaceuticals on the market today target GPCRs. However, identifying the specific GPCR target for a given compound can be a complex and challenging process, as a drug can have many interactions within the body.


GPCR Target Seeker is a tool that can streamline the early stages of drug discovery against GPCRs, providing researchers with a robust neural network that is trained to predict the primary GPCR target of any given compound. Using a multi-class classification multilayer perceptron model that was built based on methods outlined in a peer-reviewed study published in esteemed scientific journal *Nature*, this tool enables any researcher to easily screen chemical libraries and make informed decisions about which compounds to pursue for GPCR drug development. The GPCR Target Seeker tool is a valuable resource for drug discovery researchers, providing a streamlined screening method for predicting potential GPCR targets and ultimately accelerating the drug discovery process.
                ''')

def ligand_qsar():
    st.markdown('##### Enter a compound using SMILES notation')


    with st.form(key='qsar-form'):

        smi = st.text_input('Input SMILES string')
        smi = smi.strip()
        ligands = st.file_uploader('or upload file with multiple SMILES (one per line)', type='txt')
        col1, col2 = st.columns(2)
        with col1:
            submit_job = st.form_submit_button('Predict')
        with col2:
            samples = st.checkbox('Use sample SMILES file')
            if samples is True:
                with open('sample_gpcr_ligands.txt', encoding='utf-8-sig') as f:
                    ligands = [line.strip() for line in f]
            
        
        if submit_job and not ctk.valid_smiles(smi) and ligands is None:
            st.error('Invalid SMILES')
        elif submit_job and smi == '' and ligands is None:
            st.error('Enter SMILES string')
        elif submit_job and ctk.valid_smiles(smi) and ligands is None:
            prediction = qsar.deploy_model([smi])[0]
            gpcr = qsar.gpcr_encoded[prediction]
            with st.container():
                blk=ctk.makeblock(smi)
                ctk.render_mol(blk,'stick')
                st.success(f'{gpcr} - Code: {prediction}')
        elif submit_job and ligands is not None:
            if samples is not True:
                stringio = StringIO(ligands.getvalue().decode("utf-8"))
                ligands = stringio.read().split('\n')
            predictions = qsar.deploy_model(ligands)
            gpcrs = [qsar.gpcr_encoded[pred] for pred in predictions]
            pred_dict = {'SMILES': ligands, 'GPCR': gpcrs, 'Code': predictions}
            pred_df = pd.DataFrame(pred_dict)
            st.write(pred_df)

        with st.expander('Sample GPCR ligands'):
            st.write('Nicotine')
            st.code('CN1CCCC1C2=CN=CC=C2')
            st.write('Fentanyl')
            st.code('CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3')
            st.write('Tetrahydrocannabinol (THC)')
            st.code('CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O')
            st.write('5-MeO DMT')
            st.code('CN(C)CCC1=CNC2=C1C=C(C=C2)OC')
            st.write('Pramipexole')
            st.code('CCCNC1CCC2=C(C1)SC(=N2)N')

            

def model_info():
    with st.container():
        st.markdown('''
                    **Model Metrics**
                    ---
                    ''')
        with st.container():
            acc, f1, mcc = st.columns(3)
            with acc:
                st.metric('Accuracy', value ='95.7%')
            with f1:
                st.metric('F1-score', value ='95.7%')
            with mcc:
                st.metric('Matthews Correlation Coef.', value ='95.5%')

        st.markdown('''
                    Model was evaluated using stratified 5-fold cross-validation
                    ''')
        with st.container():   
            st.markdown('''
                        **Training data** 
                        ---
                        ''')
            st.markdown('''
                        **Data collection and model training** \n
                        The following spreadsheets were retrieved from the GLASS Database at https://zhanggroup.org/GLASS/.
                            
                            - Ligands.tsv
                            - GPCR_targets.tsv
                            - Interaction_actives.tsv
                            
                        ''')
            tsne_caption = '''
            All data was featurized using ECFP6 fingerprints in RDKit. Above is a 3D visualization of the model training data 
            produced through tSNE dimensionality reduction method, where each colour represents a different GPCR family. 
            Data cleaning steps were applied to the collected GPCR interaction data to produce a yield of 121,615 
            nonredundant ligands against 87 GPCR families. To reduce the scale of the dataset, the Near Miss under 
            sampling method was applied to scale the data down to 31,000 against 31 GPCR families. GPCR targets with less than 100 known ligands were removed. Encoded GPCR targets 
            the models were trained on can be seen below.
            '''
            st.image('images/tsne1.png', caption=tsne_caption)
        
        with st.container():
            st.markdown('''
                        **Implementation details**
                        ---
                        ''')
            st.write('**GPCR Targets**')
            st.write(qsar.gpcr_encoded)
            
            st.write('**Model hyperparameters:**')
            st.write(qsar.params)
            st.write('**Packages used**')
            st.write('''pandas - numpy - rdkit - scikit-learn''')
            st.markdown('''##### This machine learning model was developed using a modified version of the method described in this academic paper: https://doi.org/10.1038/s41598-021-88939-5''')




    
    
def contact():
    st.markdown('''
                This application and the neural network that powers it was solely developed by Gurveer Singh Shienh. 
                \nHe can be contacted at gsshienh@uwaterloo.ca or through LinkedIn at https://www.linkedin.com/in/gurveer-shienh/.
                ''')
    st.write('Source code for this application can be found at https://www.github.com/gurveershienh/gpcr-target-seeker')

