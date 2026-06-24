import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import os

import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import os

def load_models():
    # Menentukan path folder Models secara relatif dari posisi file ini
    base_path = os.path.join(os.path.dirname(__file__), '..', 'Models')
    
    # 1. Load Scaler dan Model Linear Regression
    ml_scaler = joblib.load(os.path.join(base_path, 'ml_scaler.joblib'))
    ann_scaler = joblib.load(os.path.join(base_path, 'ann_scaler.joblib'))
    linear_model = joblib.load(os.path.join(base_path, 'linear_regression.joblib'))
    
    # 2. Bangun Ulang Arsitektur ANN Sesuai dengan Model Anda
    ann_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(4,), name='input-layer'),
        tf.keras.layers.Dense(8, activation='relu', name='hidden-layer-1'),
        tf.keras.layers.Dropout(0.1, name='dropout'),
        tf.keras.layers.Dense(4, activation='relu', name='hidden-layer-2'),
        tf.keras.layers.Dense(1, activation='linear', name='output-layer')
    ])
    
    # 3. Muat Bobot (*Weights*) dari file .h5 yang baru Anda masukkan
    ann_model.load_weights(os.path.join(base_path, 'ann_model.h5'))
    
    return ml_scaler, ann_scaler, linear_model, ann_model

def run():
    st.title("🔮 Prediksi Kelarutan (Solubility Prediction)")
    st.write("Masukkan nilai fitur/deskriptor molekul di bawah ini untuk memprediksi nilai kelarutan.")

    # Load semua model dan scaler
    try:
        ml_scaler, ann_scaler, linear_model, ann_model = load_models()
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file di folder 'Models' sudah benar. Error: {e}")
        return

    # --- INPUT FORM ---
    st.subheader("Input Deskriptor Molekul")
    
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.number_input("MolLogP", value=0.0, format="%.4f")
        feature2 = st.number_input("MolWt", value=0.0, format="%.4f")
    with col2:
        feature3 = st.number_input("NumRotatableBonds", value=0.0, format="%.4f")
        feature4 = st.number_input("AromaticProportion", value=0.0, format="%.4f")

    # Pilih Model untuk Prediksi
    model_choice = st.selectbox("Pilih Model Prediksi", ["Linear Regression", "Artificial Neural Network (ANN)"])

    # Tombol Prediksi
    if st.button("Hitung Kelarutan"):
        # Susun data input menjadi DataFrame
        input_data = pd.DataFrame([[feature1, feature2, feature3, feature4]], 
                                   columns=['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion'])

        if model_choice == "Linear Regression":
            scaled_data = ml_scaler.transform(input_data)
            prediction = linear_model.predict(scaled_data)
            hasil_akhir = prediction[0]
        else:
            scaled_data = ann_scaler.transform(input_data)
            prediction = ann_model.predict(scaled_data, verbose=0)
            hasil_akhir = prediction[0][0]

        # --- TAMPILKAN HASIL ---
        st.markdown("---")
        st.metric(label=f"Hasil Prediksi Kelarutan ({model_choice})", value=f"{hasil_akhir:.4f} log(mol/L)")

if __name__ == "__main__":
    run()