import streamlit as st
import numpy as np
import joblib
import math
import random
st.title("prediction de la probabilité de désabonnement des clients d'un réseau mobile")
st.subheader("applicatif realisé par KRIS")
st.markdown("cette aplication utilise un modèle de ML pour prédire la probabilité de désabonnement des clients d'un réseau mobile")

#chargement du modèle
model = joblib.load( "final_model.joblib")

#definiton d'une fonction d'inférence
def inférence(regularity,montant,data_volume):
    new_data = np.array([
        regularity,montant,data_volume
    ])
    pred = model.predict(new_data.reshape(1,-1))
    return pred


   
#l'utlisateur saisie une valeur pour chaque caractéristique du client
regularity = st.number_input('REGULARITY:', value=54)
montant = st.number_input('MONTANT:', value=4250.0)
data_volume = st.number_input('DATA_VOLUME:', value=4.0)

# Création d'un bouton 'prédict' pour la prédiction du modèle
if st.button("PREDICT"):
    prediction = inférence(
        regularity, montant, data_volume
    )
    st.success(prediction)