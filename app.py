import streamlit as st
import pandas as pd
import pickle

# Muat model yang telah disimpan
with open('best_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Judul aplikasi
st.title('Prediksi Penyakit Diabetes')

# Input dari pengguna
st.header('Masukkan Data Pasien')
pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, value=0)
glucose = st.number_input('Kadar Glukosa', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('Tekanan Darah', min_value=0, max_value=122, value=70)
skin_thickness = st.number_input('Ketebalan Kulit', min_value=0, max_value=99, value=20)
insulin = st.number_input('Kadar Insulin', min_value=0, max_value=846, value=79)
bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=67.1, value=32.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.471)
age = st.number_input('Usia', min_value=21, max_value=81, value=33)

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

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi')
    if prediction[0] == 1:
        st.write('**Pasien diprediksi terkena diabetes.**')
        st.write(f'Probabilitas: {prediction_proba[0][1]*100:.2f}%')
    else:
        st.write('**Pasien diprediksi tidak terkena diabetes.**')
        st.write(f'Probabilitas: {prediction_proba[0][0]*100:.2f}%')