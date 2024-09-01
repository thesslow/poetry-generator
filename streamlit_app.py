import pickle
from joblib import load
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_poetry(seed, n_lines, model, token, seq_length, poetry_length=10):
    # Membuat dictionary inverse dari word_index
    index_word = {index: word for word, index in token.word_index.items()}
    
    generated_poetry = ""

    for i in range(n_lines):
        text = []
        for _ in range(poetry_length):
            encoded = token.texts_to_sequences([seed])[0]
            encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre')

            # Mendapatkan prediksi dari model
            y_pred = np.argmax(model.predict(encoded, verbose=0), axis=-1)[0]

            # Mendapatkan kata yang sesuai dengan indeks prediksi
            predicted_word = index_word.get(y_pred, '')

            seed = seed + ' ' + predicted_word
            text.append(predicted_word)

        # Update seed dengan baris terakhir
        seed = ' '.join(text[-seq_length:])
        # Gabungkan kata-kata menjadi teks
        text = ' '.join(text)
        generated_poetry += text + '\n'
    
    return generated_poetry.strip()

st.set_page_config(page_title="Poetry Generator", page_icon=":sparkles:", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Poetry Generator :sparkles:")

seed_text = st.text_input("Masukkan seed teks untuk memulai puisi:")
num_lines = st.slider("Jumlah baris puisi:", min_value=1, max_value=10, value=3)

poet_gen = load('poet_gen.joblib')

model = load_model('poet_gen_model.h5')
token = poet_gen['token']
seq_length = poet_gen['seq_length']

poetry_length = 7

if st.button("Generate Poetry"):
    if model and token:
        poetry = generate_poetry(seed_text, num_lines, model, token, seq_length, poetry_length=poetry_length)
        st.text_area("Generated Poetry", value=poetry, height=300)
    else:
        st.error("Model dan tokenizer belum diinisialisasi.")
