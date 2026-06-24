import pandas as pd
import streamlit as st
from joblib import load
import tensorflow as tf
from tensorflow import keras



@st.cache_data
def load_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/Rafi-Noufal12/Mini_Project_II/refs/heads/main/Datasets_Delaney_Solubility_with_Descriptors.csv"
    ).drop_duplicates().reset_index(drop=True)

@st.cache_resource
def load_model_ann():
    try:
        model_Ann = keras.models.load_model("Models/ann_model.keras")
        scaler_Ann = load("Models/ann_scaler.joblib")
        return model_Ann, scaler_Ann  # <-- Diubah agar mengembalikan 2 objek
    except Exception as e:
        st.error(f"Terdapat error pada pemanggilan ANN: {e}")
        return None, None

@st.cache_resource
def load_model_linear():
    try:
        model_linear = load("Models/linear_regression.joblib")
        scaler_linear = load("Models/ml_scaler.joblib")
        return model_linear, scaler_linear  # <-- Diubah agar mengembalikan 2 objek
    except Exception as e:
        st.error(f"Terdapat error pada pemanggilan Linear Regression: {e}")
        return None, None