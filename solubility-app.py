# import modules
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors

## Custom functions ##

# Calculate molecular descriptors
def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  AromaticAtom = sum(aa_count)
  HeavyAtom = Descriptors.HeavyAtomCount(m)
  AR = AromaticAtom/HeavyAtom
  return AR

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

# image
image = Image.open('solubility-logo.jpg')
st.image(image,use_column_width=True)

# title
st.header('Molecular Solubility Prediction Web App')

# description
st.markdown('''
This app predicts the **solubility (LogS)** values of molecules!

Data obtained from the John S. Delaney. [ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.

''')

# input molecules in sidebar

st.sidebar.header('Input SMILES')

# default input
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES input",SMILES_input)

# validate input (no empty input)
if len(SMILES)<=0:
    st.error("SMILES input must be at least one character.")
else:
    SMILES = "C\n" + SMILES # Add C as dummy first item, dummy item is needed to escape the error thrown when there is only one line of input
    SMILES = SMILES.split('\n') # removes enter

    # validate input (no empty lines and invalid character)
    accepted_inputs = ['I','F','N','O','P','S','C']
    validate_list = list( char for char in list(nested_list for nested_list in list(list(char not in accepted_inputs for char in SMILES_LINE) for SMILES_LINE in SMILES)))
    if any(len(SMILES.strip())<1 for SMILES in SMILES):
        st.error('SMILES input must not contain empty line(s).')

    elif  any(True in sublist for sublist in validate_list):
        st.error('SMILES input contains invalid character(s).')
    else:

        st.header('Input SMILES')
        SMILES[1:] # skip dummy first item

        # Calculate molecular descriptors
        st.header('Computed Molecular Descriptors')
        X = generate(SMILES) 
        X[1:] # skip dummy first item

        # read saved model
        loaded_model = pickle.load(open('solubility-model.pkl','rb'))

        # apply model to make prediction
        prediction = loaded_model.predict(X)

        # linear regression cannot predict probability

        #  output prediction
        st.header('Predicted logS value')
        st.write(prediction[1:])

