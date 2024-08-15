import streamlit as st
from streamlit_option_menu import option_menu
from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report

def casefolding(text):
    text = text.lower()
    text = text.strip()
    return text

def cleaning(text):
    text = re.sub("@[A-Za-z0-9_]+", " ", text)
    text = re.sub("#[A-Za-z0-9_]+", "", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub("[^0-9A-Za-z ]", "", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\n]', '', text)
    text = text.replace('  ', "")
    return text

stopwords_ind = stopwords.words('indonesian')
def remove_stop_words(text):
    clean_words = []
    text = text.split()
    for word in text:
        if isinstance(word, str) and word not in stopwords_ind:
            clean_words.append(word)
    return " ".join(clean_words)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
    text = stemmer.stem(text)
    return text

def tokenize(text):
    text = word_tokenize(text)
    return text

def preprocess_text(text):
    if isinstance(text, list):
        # Jika input berupa daftar token, gabungkan menjadi string
        text = ' '.join(text)

    text = casefolding(text)
    text = cleaning(text)
    text = remove_stop_words(text)
    text = stemming(text)
    text = tokenize(text)

    return text


# Fungsi untuk menyimpan dataset yang telah dilabeli
def save_labeled_dataset(dataset, dataset_name):
    dataset_path = f"uploaded_files/labeled_{dataset_name}"
    dataset.to_csv(dataset_path, index=False)
    return dataset_path

# Fungsi untuk memuat dataset yang telah dilabeli
def load_labeled_dataset(dataset_path):
    return pd.read_csv(dataset_path)

# Variabel global untuk menyimpan dataset yang telah dilabeli
labeled_datasets = {}  # Gunakan dictionary untuk menyimpan multiple datasets

upload_dir = "uploaded_files"

if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

with st.sidebar:
    selected = option_menu("Sentimen Analisis", ['Home', 'Dataset', 'Preprocessing', 'Result'],
                          icons=['house', 'database-gear', 'cpu', 'clipboard2-data'], menu_icon="cast", default_index=0)

if selected == 'Home':
    st.title('Analisis Sentimen Menggunakan Support Vector Machine ')
    st.text("""Harap Dibaca Terlebih Dahulu
1. Siapkan dataset sentimen dari twitter dengan format .csv yang 
terdiri dari dua kolom. Kolom pertama untuk sentimen berikan nama "Text" 
dan kolom kedua untuk sentimen berikan nama "Labels".

2. Pilih menu Upload Data untuk import data sentimen yang akan digunakan 
atau untuk mengganti dataset baru.

3. Lakukan text preprocessing pada menu Preprocessing.

4. Lakukan pembobotan fitur dengan memilih menu TD-IDF.

5. Lakukan pengujian analisis sentimen dengan algoritma SVM pada 
menu Klasifikasi SVM.

6. Hasil dari pengujian akan tampil pada menu Klasifikasi SVM terdiri dari 
akurasi, precision, dan recall. Confusion matrix akan tampil setelah nilai akurasi.

7. Lakukan pengujian pada model yang telah dibuat dengan memasukkan 
kalimat baru yang tidak ada di dataset dan lihat hasilnya apakah analisis 
sentimen dengan klasifikasi SVM berjalan dengan baik.""")

elif selected == 'Dataset':
    st.title('Silakan upload dataset terlebih dahulu')
    data_file = st.file_uploader("Upload CSV file", type=["csv"])

    if data_file is not None:
        file_path = f"uploaded_files/{data_file.name}"
        with open(file_path, "wb") as f:
            f.write(data_file.getvalue())

        df = pd.read_csv(file_path)
        st.dataframe(df)

        if st.button("Labelling"):
            dataset_name = data_file.name
            dataset = df.drop(['UserName','Handle','Timestamp', 'Comments', 'Likes', 'Retweets'], axis=1)
            lexicon_positive = pd.read_excel('kamus_positive.xlsx')
            lexicon_positive_dict = {}
            for index, row in lexicon_positive.iterrows():
                if row[0] not in lexicon_positive_dict:
                    lexicon_positive_dict[row[0]] = row[1]
            
            lexicon_negative = pd.read_excel('kamus_negative.xlsx')
            lexicon_negative_dict = {}
            for index, row in lexicon_negative.iterrows():
                if row[0] not in lexicon_negative_dict:
                    lexicon_negative_dict[row[0]] = row[1]

            def sentiment_analysis_lexicon_indonesia(text):
                text = str(text)
                score = 0
                for word in text.split():
                    if isinstance(word, str):  # Memeriksa apakah kata adalah string
                        if word.lower() in lexicon_positive_dict:
                            score += lexicon_positive_dict[word.lower()]
                        elif word.lower() in lexicon_negative_dict:
                            score += lexicon_negative_dict[word.lower()]

                sentimen = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
                return score, sentimen


            results = dataset['Text'].apply(sentiment_analysis_lexicon_indonesia)
            results = list(zip(*results))
            dataset['Polarity Score'] = results[0]
            dataset['Labels'] = results[1]

            labeled_datasets[dataset_name] = dataset
            labeled_dataset_path = save_labeled_dataset(dataset, dataset_name)

            st.subheader(f'Dataset Setelah Labelling: {dataset_name}')
            st.dataframe(labeled_datasets[dataset_name])

            st.session_state.data_file_path = file_path
            st.session_state.labeled_dataset_path = labeled_dataset_path

elif selected == 'Preprocessing':
    st.title('Text Preprocessing')

    # Inisialisasi labeled_dataset_preprocessed_path
    if 'labeled_dataset_preprocessed_path' not in st.session_state:
        st.session_state.labeled_dataset_preprocessed_path = None

    if 'labeled_dataset_path' not in st.session_state:
        st.warning('Silakan upload dataset dan lakukan labelling terlebih dahulu.')
    else:
        # Load dataset dari path yang telah disimpan
        labeled_dataset_path = st.session_state.labeled_dataset_path
        labeled_dataset = load_labeled_dataset(labeled_dataset_path)
        
        # Menampilkan data yang telah dilabeli
        st.subheader('Dataset yang Telah Dilabeli')
        st.dataframe(labeled_dataset)
        
        preprocessing_done = False  # Variabel status preprocessing
        if st.button('Proses Preprocessing'):
            # Menambahkan kolom untuk hasil preprocessing
            labeled_dataset['Case_Folding'] = labeled_dataset['Text'].fillna('').apply(lambda x: casefolding(x))
            labeled_dataset['Cleaning_Text'] = labeled_dataset['Case_Folding'].fillna('').apply(lambda x: cleaning(x))
            labeled_dataset['Remove_Text'] = labeled_dataset['Cleaning_Text'].fillna('').apply(lambda x: remove_stop_words(x))
            labeled_dataset['Stemming_Text'] = labeled_dataset['Remove_Text'].fillna('').apply(lambda x: stemming(x))
            labeled_dataset['Tokenize_Text'] = labeled_dataset['Stemming_Text'].fillna('').apply(lambda x: tokenize(x))
            labeled_dataset['Processed_Text'] = labeled_dataset['Text'].fillna('').apply(lambda x: preprocess_text(x))
            
            # Simpan dataset setelah preprocessing
            labeled_dataset_preprocessed = labeled_dataset.copy()
            
            # Simpan hasil preprocessing ke st.session_state
            st.session_state.labeled_dataset_preprocessed = labeled_dataset_preprocessed
            labeled_dataset_preprocessed_path = save_labeled_dataset(labeled_dataset_preprocessed, "preprocessed_" + os.path.basename(labeled_dataset_path))
            st.session_state.labeled_dataset_preprocessed_path = labeled_dataset_preprocessed_path  # Inisialisasi path
            st.subheader('Data Setelah Preprocessing')
            st.dataframe(labeled_dataset_preprocessed[['Text','Case_Folding','Cleaning_Text', 'Remove_Text','Stemming_Text','Tokenize_Text','Processed_Text']])
            preprocessing_done = True  # Setelah preprocessing selesai
            st.success('Proses preprocessing selesai.')

elif selected == 'Result':
    st.title('Hasil Klasifikasi SVM')
    
    if 'data_file_path' not in st.session_state:
        st.warning('Mohon unggah dataset terlebih dahulu.')
    else:
        df = pd.read_csv(st.session_state.data_file_path)
        st.subheader('Data Setelah Preprocessing dan Labeling')  
        st.dataframe(df[['Hasil_Preprocessing','Polarity Score','Labels']])  

    if st.button('Proses TF-IDF dan SVM'):
        # Filter dokumen yang tidak kosong setelah preprocessing
        non_empty_documents = df['Hasil_Preprocessing'].fillna('')

        if not non_empty_documents.any():
            st.warning("Semua dokumen menjadi kosong setelah preprocessing. Periksa langkah-langkah preprocessing Anda.")
        else:
            # Lakukan pembobotan fitur TF-IDF
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(non_empty_documents)
            Y = df['Labels'].map({'Positive': 1, 'Negative': 0})

            # Buat DataFrame dari matriks TF-IDF
            tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

            # Tampilkan DataFrame hasil TF-IDF
            st.subheader('Hasil TF-IDF')
            st.dataframe(tfidf_df)

            # Lakukan pembagian data menjadi data latih dan uji
            X_train, X_test, Y_train, Y_test = train_test_split(df['Hasil_Preprocessing'],
                                                                df['Labels'], test_size=0.2,
                                                                stratify=df['Labels'], random_state=42)
            # Terapkan preprocessing pada data uji
            X_test = X_test.apply(preprocess_text)
            
            # Lakukan klasifikasi SVM
            clfsvm = svm.SVC(kernel="linear")
            clfsvm.fit(X_train, Y_train)
            predict = clfsvm.predict(X_test)

            st.write("SVM Accuracy score  -> ", accuracy_score(predict, Y_test) * 100)
            st.write("SVM Recall Score    -> ", recall_score(predict, Y_test, average='macro') * 100)
            st.write("SVM Precision score -> ", precision_score(predict, Y_test, average='macro') * 100)
            st.write("SVM f1 score        -> ", f1_score(predict, Y_test, average='macro') * 100)
            st.write("===========================================================")
            st.write('Confusion matrix : \n', confusion_matrix(predict, Y_test))
            st.write("===========================================================")
            st.text('Classification report : \n' + classification_report(predict, Y_test, zero_division=0))
            st.write("===========================================================")



# Jalankan aplikasi jika dijalankan sebagai script utama
if __name__ == '__main__':
    pass
