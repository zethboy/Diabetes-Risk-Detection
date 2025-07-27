import streamlit as st
import pandas as pd
import pickle

# Tambahkan CSS custom untuk mempercantik tampilan
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        margin-top: 10px;
        box-shadow: 0 2px 8px rgba(76,175,80,0.08);
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #43a047;
    }
    /* Input number lebih terang dan jelas */
    input[type="number"] {
        background-color: #fff !important;
        color: #222 !important;
        border: 1.5px solid #b0bec5 !important;
        border-radius: 8px !important;
        font-size: 1.1em !important;
        padding: 0.7em 0.6em 0.7em 0.6em !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 1.1em !important;
        width: 100% !important;
        box-sizing: border-box !important;
        text-align: right !important;
        display: flex;
        align-items: center;
        min-height: 2.7em;
    }
    input[type="number"]:focus {
        border: 2px solid #4CAF50 !important;
        outline: none !important;
        background: #f1f8e9 !important;
    }
    /* Hilangkan panah input number di Chrome, Safari, Edge */
    input[type=number]::-webkit-inner-spin-button, 
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    /* Hilangkan panah input number di Firefox */
    input[type=number] {
        -moz-appearance: textfield;
    }
    .st-bb {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(44,62,80,0.07);
        padding: 2em 2em 1em 2em;
        margin-bottom: 2em;
        border: 1.5px solid #e3e6ea;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Muat model yang telah disimpan
with open('best_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Judul aplikasi

# Judul aplikasi dengan emoji
st.markdown('<h1 style="color:#2e4053;">🩺 Prediksi Penyakit <span style="color:#4CAF50;">Diabetes</span></h1>', unsafe_allow_html=True)

# Input dari pengguna

st.markdown('<div class="st-bb">', unsafe_allow_html=True)
st.header('📝 Masukkan Data Pasien')
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Kadar Glukosa', min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input('Tekanan Darah', min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input('Ketebalan Kulit', min_value=0, max_value=99, value=20)
with col2:
    insulin = st.number_input('Kadar Insulin', min_value=0, max_value=846, value=79)
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=67.1, value=32.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.471)
    age = st.number_input('Usia', min_value=21, max_value=81, value=33)
st.markdown('</div>', unsafe_allow_html=True)

# Tombol untuk membuat prediksi
if st.button('Prediksi'):
    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Buat prediksi
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.markdown('<div class="st-bb">', unsafe_allow_html=True)
    st.subheader('🔎 Hasil Prediksi')
    if prediction[0] == 1:
        st.error('**Pasien diprediksi terkena diabetes.**')
        st.write(f'🩸 <b>Probabilitas:</b> <span style="color:#c0392b; font-size:1.2em;">{prediction_proba[0][1]*100:.2f}%</span>', unsafe_allow_html=True)
    else:
        st.success('**Pasien diprediksi tidak terkena diabetes.**')
        st.write(f'💚 <b>Probabilitas:</b> <span style="color:#229954; font-size:1.2em;">{prediction_proba[0][0]*100:.2f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)