import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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
    text = re.sub("[^0-9A-Za-z ]", "", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
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
        if word not in stopwords_ind:
            clean_words.append(word)
    return " ".join(clean_words)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
    text = stemmer.stem(text)
    return text

def preprocess_text(text):
    text = casefolding(text)
    text = cleaning(text)
    text = remove_stop_words(text)
    text = stemming(text)
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
1. Siapkan dataset sentimen dari Twitter dengan format .csv yang 
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
            dataset = df.drop(['Timestamp', 'Comments', 'Likes', 'Retweets'], axis=1)
            labeled_datasets[dataset_name] = dataset
            labeled_dataset_path = save_labeled_dataset(dataset, dataset_name)

            st.subheader(f'Dataset Setelah Labelling: {dataset_name}')
            st.dataframe(labeled_datasets[dataset_name])

            st.session_state.data_file_path = file_path
            st.session_state.labeled_dataset_path = labeled_dataset_path

elif selected == 'Preprocessing':
    st.title('Text Preprocessing')

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
            labeled_dataset['Processed_Text'] = labeled_dataset['Text'].fillna('').apply(lambda x: preprocess_text(x))
            
            # Simpan dataset setelah preprocessing
            labeled_dataset_preprocessed = labeled_dataset.copy()
            
            # Simpan hasil preprocessing ke st.session_state
            st.session_state.labeled_dataset_preprocessed = labeled_dataset_preprocessed
            labeled_dataset_preprocessed_path = save_labeled_dataset(labeled_dataset_preprocessed, "preprocessed_" + os.path.basename(labeled_dataset_path))
            st.session_state.labeled_dataset_preprocessed_path = labeled_dataset_preprocessed_path  # Inisialisasi path
            st.subheader('Data Setelah Preprocessing')
            st.dataframe(labeled_dataset_preprocessed[['Text', 'Processed_Text']])
            preprocessing_done = True  # Setelah preprocessing selesai
            st.success('Proses preprocessing selesai.')

elif selected == 'Result':
    st.title('Hasil Klasifikasi SVM')
    if 'labeled_dataset_preprocessed' in st.session_state:
        labeled_dataset_preprocessed = st.session_state.labeled_dataset_preprocessed

        # Pastikan ketersediaan labeled_dataset_preprocessed_path
        if 'labeled_dataset_preprocessed_path' in st.session_state:
            labeled_dataset_preprocessed_path = st.session_state.labeled_dataset_preprocessed_path

            st.subheader('Dataset yang Telah Dilabeli')
            st.dataframe(labeled_dataset_preprocessed)

            if st.button('Proses TF-IDF'):
                # Filter dokumen yang tidak kosong setelah preprocessing
                non_empty_documents = labeled_dataset_preprocessed['Processed_Text'].dropna()

                # Lakukan pembagian data menjadi data latih dan uji
                X_train, X_test, Y_train, Y_test = train_test_split(non_empty_documents,
                                                                    labeled_dataset_preprocessed['Labels'], test_size=0.2,
                                                                    stratify=labeled_dataset_preprocessed['Labels'], random_state=42)

                label_mapping = {'Positive': 1, 'Negative': 0}
                Y_train = Y_train.map(label_mapping)
                Y_test = Y_test.map(label_mapping)

                vectorizer = TfidfVectorizer()
                X_train = vectorizer.fit_transform(X_train)
                X_test = vectorizer.transform(X_test)
                
                # Lakukan pengujian SVM menggunakan labeled_dataset_preprocessed
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
        else:
            st.warning("Mohon lakukan preprocessing terlebih dahulu.")
    else:
        st.warning("Mohon upload dataset dan lakukan labelling terlebih dahulu.")
