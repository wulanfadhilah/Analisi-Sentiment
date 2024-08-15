import streamlit as st
from streamlit_option_menu import option_menu
from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np
import re
import string
import nltk
import os
import time
import csv
import pickle
import joblib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.exceptions import NotFittedError
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report

def get_tweet_data(card):
    """Extract data from tweet card"""
    username = card.find_element(By.XPATH, './/span').text
    try:
        handle = card.find_element(By.XPATH, './/span[contains(text(), "@")]').text
    except NoSuchElementException:
        return
    
    try:
        postdate = card.find_element(By.XPATH, './/time').get_attribute('datetime')
    except NoSuchElementException:
        return
    
    text = card.find_element(By.XPATH, './/div[@data-testid="tweetText"]').text
    reply_cnt = card.find_element(By.XPATH, './/div[@data-testid="reply"]').text
    retweet_cnt = card.find_element(By.XPATH, './/div[@data-testid="retweet"]').text
    like_cnt = card.find_element(By.XPATH, './/div[@data-testid="like"]').text
    
    tweet = (username, handle, postdate, text, reply_cnt, retweet_cnt, like_cnt)
    return tweet 



def casefolding(text):
    text = text.lower()
    text = text.strip()
    return text

def cleaning(text):
  text = re.sub("@[A-Za-z0-9_]+"," ", text)
  text = re.sub("#[A-Za-z0-9_]+"," ", text)
  text = re.sub(r'http://\S+|www\.\S+', '', text)
  text = re.sub("[^0-9A-Za-z ]", " " , text)
  text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", " ", text)
  text = re.sub(r'[-+]?[0-9]+', '', text)
  text = re.sub(r'[^\w\s]','', text)
  text = re.sub(r'[\n]','', text)
  text = re.sub(r'\d+', '', text)
  text = text.replace('  '," ")
  return text

col_names_abbeviation = ['before','after']
indo_abbreviation = pd.read_csv('kamus_singkatan.csv', delimiter=';', names=col_names_abbeviation)
indo_abbreviation.head()

def replace_abbreviations(text, abbreviation):
    # Buat dictionary dari dataframe indo_abbreviation
    abbreviation_dict = dict(zip(abbreviation['before'], abbreviation['after']))

    # Fungsi untuk mengganti kata dalam text
    def replace_words(text):
        words = text.split()
        new_words = [abbreviation_dict[word] if word in abbreviation_dict else word for word in words]
        return ' '.join(new_words)

    # Terapkan fungsi replace_words ke text
    text = replace_words(text)

    return text

col_names_slank_words = ['before','after']
indo_slank_words = pd.read_csv('kamusalay.csv', delimiter=',', names=col_names_slank_words)
indo_slank_words.head()

def replace_slank_words(text, slank_words):
    # Buat dictionary dari dataframe indo_abbreviation
    abbreviation_dict = dict(zip(slank_words['before'], slank_words['after']))

    # Fungsi untuk mengganti kata dalam text
    def replace_words(text):
        words = text.split()
        new_words = [abbreviation_dict[word] if word in abbreviation_dict else word for word in words]
        return ' '.join(new_words)

    # Terapkan fungsi replace_words ke setiap text
    text = replace_words(text)

    return text

stopwords_ind = stopwords.words('indonesian')
df_stopwords = pd.read_csv('short_word.csv')
more_stopwords = df_stopwords['short_words'].tolist()
stopwords_ind = stopwords_ind + more_stopwords
df_stopwords_combined = pd.DataFrame({'stopwords': stopwords_ind})
df_stopwords_combined.to_csv('list_stopwords.csv', index=False)
preserved_words = ["tidak", "jangan"]
def remove_stop_words(text, preserved_words=None):
    clean_words = []
    text = text.split()
    for word in text:
        # Periksa apakah kata ada dalam daftar kata yang ingin dipertahankan
        if preserved_words and word in preserved_words:
            clean_words.append(word)
        elif word not in stopwords_ind:
            clean_words.append(word)
    return " ".join(clean_words)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
custom_exceptions = {'pemilu'}

def stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) if word not in custom_exceptions else word for word in words]
    result = ' '.join(stemmed_words)
    return result

def tokenize(text):
    text = word_tokenize(text)
    return text

def remove_whitespace_and_combine(text, tokens):
    if tokens and isinstance(tokens, list):  # Periksa apakah tokens bukan None dan merupakan list
        # Menghapus whitespace yang tidak perlu
        tokens = [token.strip() for token in tokens if token and isinstance(token, str) and token.strip()]

        # Menggabungkan kembali kata-kata menjadi kalimat yang padu
        result = ' '.join(tokens)
        return result
    else:
        return None

def preprocess_text(text):
    if isinstance(text, list):
        # Jika input berupa daftar token, gabungkan menjadi string
        text = ' '.join(text)

    if text is not None:  # Tambahkan pengecekan untuk nilai None
        text = casefolding(text)
        text = cleaning(text)
        text = replace_abbreviations(text, indo_abbreviation)  # Menyertakan argumen 'abbreviation'
        text = replace_slank_words(text, indo_slank_words)    # Menyertakan argumen 'slank_words'
        text = remove_stop_words(text)
        text = stemming(text)
        text = ' '.join(tokenize(text))  # Ubah hasil tokenisasi menjadi string
        
        return text
    else:
        return None

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
    st.session_state.page = option_menu("Sentimen Analisis SVM", ['Home','Scraping', 'Dataset', 'Preprocessing', 'Result'],
                          icons=['house','', 'database-gear', 'cpu', 'clipboard2-data'], menu_icon="cast", default_index=0)

if st.session_state.page == 'Home':
    st.title('Analisis Sentimen Menggunakan Support Vector Machine ')
    st.text("""Harap Dibaca Terlebih Dahulu
1. Siapkan dataset sentimen dari twitter dengan format .csv
Jika belum ada dataset, lakukan scraping data terlebih dahulu melalui menu Scraping

2. Pilih menu Upload Data untuk import data sentimen yang akan digunakan 
atau untuk mengganti dataset baru.

3. Lakukan proses text preprocessing dan labelling data pada menu Preprocessing.

4. Lakukan proses TF-IDF dan pengujian analisis sentimen dengan algoritma SVM pada
menu Result.

5. Hasil dari pengujian akan tampil pada menu Klasifikasi SVM terdiri dari 
akurasi, precision, dan recall. Confusion matrix akan tampil setelah nilai akurasi.
""")

elif st.session_state.page == 'Scraping':
    st.title("Scraping Data Twitter")

    # Input untuk akun Twitter, topik pencarian, dan nama dataset
    Acc_Username = st.text_input("Masukkan username Twitter:")
    Acc_Password = st.text_input("Masukkan password Twitter:")
    Tweets_Query = st.text_input("Masukkan topik pencarian Twitter:")
    dataset_name = st.text_input("Masukkan nama untuk dataset:")

    # Tombol untuk memulai scraping
    if st.button("Mulai Scraping"):
        # Periksa apakah akun Twitter, topik pencarian, dan nama dataset sudah dimasukkan
        if not Acc_Username or not Tweets_Query:
            st.warning("Silakan masukkan username Twitter, topik pencarian, dan nama dataset terlebih dahulu.")
        else:
            # Inisialisasi WebDriver (ganti PATH dengan lokasi WebDriver di komputer Anda)
            driver = webdriver.Edge() # Gunakan driver yang sesuai untuk browser Anda
            driver.get('https://twitter.com/login')

            username_input = WebDriverWait(driver, timeout=10).until(
                EC.visibility_of_element_located((By.XPATH, '//input'))
            )
            username_input.send_keys(Acc_Username)
            next_btn = driver.find_element(By.XPATH, '//*[text()="Next"]')
            next_btn.click()

            # Tunggu untuk input password muncul
            password_input = WebDriverWait(driver, timeout=10).until(
                EC.visibility_of_element_located((By.XPATH, '//input[@name="password"]')))
            password_input.send_keys(Acc_Password)

            login_btn = driver.find_element(By.XPATH, '//*[text()="Log in"]')
            login_btn.click()
            
            time.sleep(5)

            # Cari elemen setelah penundaan
            explore_btn = driver.find_element(By.XPATH, '//a[@href="/explore"]')
            explore_btn.click()

            explore_input = WebDriverWait(driver, timeout=50).until(
                EC.visibility_of_element_located((By.XPATH, '//input'))
            )
            explore_input.send_keys(Tweets_Query)
            explore_input.send_keys(Keys.ENTER)

            latest_btn = WebDriverWait(driver, 50).until(
                EC.presence_of_element_located((By.XPATH, '//div[@data-testid][2]/div/div[2]'))
            )
            latest_btn.click()
            
            data = []
            tweet_ids = set()

            last_position = driver.execute_script("return window.pageYOffset;")
            scrolling = True
            wait = WebDriverWait(driver, 50)
            while scrolling:
                page_cards = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
                for card in page_cards[-15:]:
                    tweet = get_tweet_data(card)
                    if tweet:
                        tweet_id = ''.join(tweet)
                        if tweet_id not in tweet_ids:
                            tweet_ids.add(tweet_id)
                            data.append(tweet)
                        
                scroll_attempt = 0
                while True:
                    # cek posisi scroll
                    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                    time.sleep(3)
                    curr_position = driver.execute_script("return window.pageYOffset;")
                    if last_position == curr_position:
                        scroll_attempt += 1
                        
                        # akhir dari daerah scroll
                        if scroll_attempt >= 3:
                            scrolling = False
                            break
                        else:
                            time.sleep(2) # coba scroll lagi
                    else:
                        last_position = curr_position
                        break

           # Pastikan folder "Dataset" ada
            folder_name = 'Dataset'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Simpan dataset dengan nama yang ditentukan pengguna
            if dataset_name:
                file_name = f"{dataset_name}.csv"
                file_path = os.path.join(folder_name, file_name)

                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    header = ['UserName', 'Handle', 'Timestamp', 'Text', 'Comments', 'Likes', 'Retweets']
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data)

                st.success(f"Scraping data Twitter selesai. Data disimpan di dalam folder 'Dataset' dengan nama '{file_name}'.")


elif st.session_state.page == 'Dataset':
    st.title('Silakan upload dataset terlebih dahulu')
    data_file = st.file_uploader("Upload file CSV", type=["csv"])

    if data_file is not None:
        # Baca DataFrame dari file CSV yang diunggah
        df = pd.read_csv(data_file)
        
        # Simpan DataFrame ke dalam session state
        st.session_state.uploaded_data = df

    # Periksa apakah ada data yang diunggah di session state
    if 'uploaded_data' in st.session_state:
        st.title('Data yang Telah Diunggah')
        df = st.session_state.uploaded_data
        st.dataframe(df)
    


elif st.session_state.page == 'Preprocessing':
    st.title('Text Preprocessing')
    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        st.subheader('Data Sebelum Preprocessing')
        st.dataframe(df)
    else:
        st.warning('Mohon unggah dataset terlebih dahulu.')

    if st.button('Proses Preprocessing'):
        dataset_name = st.session_state.uploaded_data
        df = df.drop(['UserName','Handle','Timestamp', 'Comments', 'Likes', 'Retweets'], axis=1)
        df['Case_Folding'] = df['Text'].fillna('').apply(lambda x: casefolding(x))
        df['Cleaning_Text'] = df['Case_Folding'].fillna('').apply(lambda x: cleaning(x))
        df['Normalisasi_Text'] = df['Cleaning_Text'].apply(replace_abbreviations, abbreviation=indo_abbreviation)
        df['Formalisasi_Text'] = df['Normalisasi_Text'].apply(replace_slank_words, slank_words=indo_slank_words)
        df['Remove_Text'] = df['Formalisasi_Text'].fillna('').apply(lambda x: remove_stop_words(x))
        df['Stemming_Text'] = df['Remove_Text'].fillna('').apply(lambda x: stemming(x))
        df['Tokenize_Text'] = df['Stemming_Text'].fillna('').apply(lambda x: tokenize(x))
        df['Processed_Text'] = df['Text'].fillna('').apply(lambda x: preprocess_text(x))
        
        st.session_state.preprocessed_data = df  # Simpan dataframe hasil preprocessing
        st.dataframe(df[['Text','Case_Folding','Cleaning_Text','Normalisasi_Text','Formalisasi_Text','Remove_Text','Stemming_Text','Tokenize_Text','Processed_Text']])

        label_column = 'Processed_Text'  # Mengganti label kolom dengan hasil preprocessing
        
        # Lakukan proses labelling dari kolom 'Hasil_Preprocessing'
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
                if isinstance(word, str): 
                    if word.lower() in lexicon_positive_dict:
                        score += lexicon_positive_dict[word.lower()]
                    elif word.lower() in lexicon_negative_dict:
                        score += lexicon_negative_dict[word.lower()]

            sentimen = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
            return score, sentimen
        
        results = df['Processed_Text'].apply(sentiment_analysis_lexicon_indonesia)
        results = list(zip(*results))
        df['Polarity Score'] = results[0]
        df['Labels'] = results[1]
        
        st.subheader('Data Setelah Preprocessing dan Labeling')
        st.dataframe(df[['Processed_Text','Polarity Score','Labels']])
        
        # Simpan dataframe hasil preprocessing dan labeling di session state
        st.session_state.preprocessed_data = df
        
        labeled_dataset_path = save_labeled_dataset(df, "labeled_dataset.csv")
        st.success('Proses preprocessing dan labeling selesai.')
        st.success(f'Dataset yang sudah diproses dan dilabeli disimpan di: {labeled_dataset_path}')
        
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    
    if st.session_state.preprocessed_data is not None:
        st.subheader('Data Setelah Preprocessing dan Labeling')
        st.dataframe(st.session_state.preprocessed_data)

            
# Mendefinisikan halaman Result
if st.session_state.page == 'Result':
    st.title('Hasil Klasifikasi SVM')
    
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    
    if st.session_state.preprocessed_data is None:
        st.warning('Mohon lakukan preprocessing terlebih dahulu.')
    else:
        df = st.session_state.preprocessed_data
        st.subheader('Data Setelah Preprocessing dan Labeling')  
        st.dataframe(df[['Processed_Text','Labels']])  

    if st.button('Proses TF-IDF dan SVM'):
        # Filter dokumen yang tidak kosong setelah preprocessing
        non_empty_documents = df['Processed_Text'].fillna('')  # Mengisi nilai NaN dengan string kosong

        if non_empty_documents.empty:  # Menggunakan empty untuk memeriksa apakah DataFrame kosong
            st.warning("Semua dokumen menjadi kosong setelah preprocessing. Periksa langkah-langkah preprocessing Anda.")
        else:
            # Hapus baris yang mengandung nilai NaN dari data target Y
            df = df.dropna(subset=['Labels'])

            # Lakukan pembobotan fitur TF-IDF pada data pelatihan
            st.session_state.vectorizer = TfidfVectorizer()
            X_train = st.session_state.vectorizer.fit_transform(non_empty_documents)
            Y_train = df['Labels'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

            # Buat DataFrame dari matriks TF-IDF
            tfidf_df = pd.DataFrame(X_train.toarray(), columns=st.session_state.vectorizer.get_feature_names_out())

            # Tampilkan DataFrame hasil TF-IDF
            st.subheader('Hasil TF-IDF')
            st.dataframe(tfidf_df)
            st.write("===========================================================")
            
            # Lakukan pembagian data menjadi data latih dan uji
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train, random_state=42)
            
               # Hitung nilai decision function

            # Lakukan klasifikasi SVM
            clfsvm = svm.SVC(kernel="linear")
            clfsvm.fit(X_train, Y_train)
            predict = clfsvm.predict(X_test)
            predict_labels = ['positif' if p == 1 else 'negatif' if p == -1 else 'netral' for p in predict]
            
            decision_function_values = clfsvm.decision_function(X_test)

           
            # Buat DataFrame untuk menampung hasil
            df_decision_function = pd.DataFrame(decision_function_values, columns=['Decision Value for Class Negatif', 'Decision Value for Class Netral', 'Decision Value for Class Positif'])

            # Tampilkan DataFrame dalam bentuk tabel di Streamlit
            st.write("Tabel Nilai Decision Function:")
            st.write(df_decision_function)

            # st.write("Heatmap Decision Function:")
            # plt.figure(figsize=(20, 10))
            # sns.heatmap(decision_function_values, cmap='coolwarm', annot=True, fmt=".2f")
            # st.pyplot(plt)
            
            # Tampilkan hasil prediksi dalam bentuk matriks
           
           # Create DataFrame with predictions and index
            df_predictions = pd.DataFrame({'Data Ke': range(1, len(predict_labels) + 1), 'Prediksi': predict_labels})

            # Display DataFrame
            st.write("Hasil Prediksi dengan SVM:")
            st.write(df_predictions)
            
            # Data jumlah prediksi per kelas
            positif_count = sum(predict == 1)
            negatif_count = sum(predict == -1)
            netral_count = sum(predict == 0)
            labels = ['Positif', 'Negatif', 'Netral']
            sizes = [positif_count, negatif_count, netral_count]
            colors = ['#ff9999','#66b3ff','#99ff99']

            # Buat pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sizes, labels=[f'{label}\n({size})' for label, size in zip(labels, sizes)], colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 8})
            ax.set_title('Jumlah Prediksi per Kelas', fontsize=10)
            st.pyplot(fig)
            
            prediksi_benar = (predict == Y_test).sum()
            prediksi_salah = (predict != Y_test).sum()

            st.write('Jumlah prediksi benar\t:', prediksi_benar)
            st.write('Jumlah prediksi salah\t:', prediksi_salah)

            accuracy = prediksi_benar / (prediksi_benar + prediksi_salah)*100
            st.write('Akurasi pengujian\t:', accuracy, '%')
            
            
            st.write("===========================================================")
            # Hitung skor klasifikasi
            accuracy = accuracy_score(predict, Y_test) * 100
            recall = recall_score(predict, Y_test, average='macro') * 100
            precision = precision_score(predict, Y_test, average='macro') * 100
            f1 = f1_score(predict, Y_test, average='macro') * 100

            # Simpan skor klasifikasi dalam DataFrame
            skor_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score'],
                'Score': [accuracy, recall, precision, f1]
            })

            # Tampilkan skor klasifikasi sebagai DataFrame
            st.subheader('Skor Klasifikasi')
            st.dataframe(skor_df)
            
            st.write("===========================================================")
            st.subheader('Confusion Matrix')
            # Hitung confusion matrix
            cm = confusion_matrix(predict, Y_test)

            # Plot confusion matrix
            plt.figure(figsize=(8, 5))
            sns.heatmap(cm, annot=True, fmt=".0f")
            plt.xlabel("Prediksi")
            plt.ylabel("True")
            file_path = "confusion_matrix.png"  # Atur nama file dan path sesuai dengan kebutuhan Anda
            plt.savefig(file_path)


            # Tampilkan plot dalam aplikasi Streamlit
            st.pyplot(plt.gcf())
            
            # Calculate classification report
            report = classification_report(predict, Y_test, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            # Display the classification report as a DataFrame
            st.write("===========================================================")
            st.subheader('Classification Report')
            st.dataframe(report_df)
            
        st.session_state.clfsvm = clfsvm
        # st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_df = tfidf_df
        st.session_state.df_decision_function = df_decision_function
        st.session_state.df_predictions = df_predictions
        st.session_state.fig = fig
        st.session_state.prediksi_benar = prediksi_benar
        st.session_state.prediksi_salah = prediksi_salah
        st.session_state.accuracy = accuracy
        st.session_state.skor_df = skor_df
        cm = confusion_matrix(predict, Y_test)
        st.session_state.cm = cm
        st.session_state.report_df = report_df
        


    new_text = st.text_area("Masukkan teks baru untuk diprediksi:", "")
        
    if st.button('Prediksi Sentimen'):
        if 'clfsvm' in st.session_state and 'vectorizer' in st.session_state:
            clfsvm = st.session_state.clfsvm
            vectorizer = st.session_state.vectorizer  # Gunakan vectorizer dari session state

        if not new_text:
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            new_text_processed = preprocess_text(new_text)  # Lakukan preprocessing teks baru

            if new_text_processed:
                try:
                    # Lakukan pembobotan fitur TF-IDF menggunakan vectorizer yang telah dilatih sebelumnya
                    new_text_vectorized = vectorizer.transform([new_text_processed])

                    # Tampilkan teks yang sudah diproses
                    st.text_area("Teks Setelah Preprocessing:", new_text_processed)
                    # Tampilkan hasil TF-IDF
                    # st.write("Hasil TF-IDF:")
                    # df_tfidf = pd.DataFrame(new_text_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
                    # st.write(df_tfidf)
                    
                        # Lakukan prediksi sentimen
                    prediction = clfsvm.predict(new_text_vectorized)
                    
                    decision_function_values = clfsvm.decision_function(new_text_vectorized)

        
                    # Buat DataFrame untuk menampung hasil
                    df_decision_function = pd.DataFrame(decision_function_values, columns=['Decision Value for Class Negatif', 'Decision Value for Class Netral', 'Decision Value for Class Positif'])

                    # Tampilkan DataFrame dalam bentuk tabel di Streamlit
                    st.write("Tabel Nilai Decision Function:")
                    st.write(df_decision_function)


                    if prediction == 1:
                        sentimen = 'Positif'
                    elif prediction == -1:
                        sentimen = 'Negatif'
                    else:
                        sentimen = 'Netral'

                    st.success(f"Prediksi sentimen adalah: {sentimen}")
                    
                except NotFittedError:
                    st.warning("TF-IDF vectorizer belum difit. Silakan lakukan proses TF-IDF dan SVM terlebih dahulu.")
            else:
                st.warning("Teks yang dimasukkan tidak valid setelah preprocessing.")
                
    if st.button('Tampilkan Proses Sebelumnya'):
        # tambahkan kode untuk menampilkan hasil prediksi di sini
        if 'tfidf_df' not in st.session_state:
            st.session_state.tfidf_df = None
        if st.session_state.tfidf_df is not None:
            st.subheader('Hasil TF-IDF')
            st.dataframe(st.session_state.tfidf_df)
            
        if 'df_predictions' not in st.session_state:
            st.session_state.df_predictions = None
        if st.session_state.df_predictions is not None:
            st.write("Hasil Prediksi dengan SVM:")
            st.write(st.session_state.df_predictions)

        # tambahkan kode untuk menampilkan visualisasi di sini
        if 'fig' not in st.session_state:
            st.session_state.fig = None
        if st.session_state.fig is not None:
            st.pyplot(st.session_state.fig)

        # tambahkan kode untuk menampilkan metrik evaluasi di sini
        if 'prediksi_benar' not in st.session_state:
            st.session_state.prediksi_benar = None
        if st.session_state.prediksi_benar is not None:
            st.write('Jumlah prediksi benar\t:', st.session_state.prediksi_benar)

        # tambahkan kode untuk menampilkan metrik evaluasi di sini
        if 'prediksi_salah' not in st.session_state:
            st.session_state.prediksi_salah = None
        if st.session_state.prediksi_salah is not None:
            st.write('Jumlah prediksi salah\t:', st.session_state.prediksi_salah)

        # tambahkan kode untuk menampilkan metrik evaluasi di sini
        if 'accuracy' not in st.session_state:
            st.session_state.accuracy = None
        if st.session_state.accuracy is not None:
            st.write('Akurasi pengujian\t:', st.session_state.accuracy, '%')

        # tambahkan kode untuk menampilkan metrik evaluasi di sini
        if 'skor_df' not in st.session_state:
            st.session_state.skor_df = None
        if st.session_state.skor_df is not None:
            st.subheader('Skor Klasifikasi')
            st.dataframe(st.session_state.skor_df)

        # tambahkan kode untuk menampilkan visualisasi di sini
        if 'cm' not in st.session_state:
            st.session_state.cm = None
        if st.session_state.cm is not None:
            print("Confusion matrix:", st.session_state.cm)
            st.write("Confusion Matrix")
            st.image("confusion_matrix.png")

        # tambahkan kode untuk menampilkan laporan klasifikasi di sini
        if 'report_df' not in st.session_state:
            st.session_state.report_df = None
        if st.session_state.report_df is not None:
            st.write("Classification Report")
            st.dataframe(st.session_state.report_df)         
    else:
        st.warning('Mohon lakukan proses TF-IDF dan SVM terlebih dahulu sebelum melakukan prediksi.')
    



if __name__ == '__main__':
    pass
