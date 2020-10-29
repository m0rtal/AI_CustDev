import pandas as pd
from catboost.text_processing import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.cluster import DBSCAN
from rutermextract import TermExtractor

# nltk.download("stop_words")

stop_words = list(nltk.corpus.stopwords.words("russian"))
stop_words.extend(["достоинства", "недостатки", "комментарий"])


def tokenize_list(lst):
    '''
    Принимает на вход список предложений
    Возвращает массив (список списков) очищенных данных - без пунктаации и в нижнем регистре
    '''
    return [Tokenizer(separator_type='BySense', token_types=['Word'], languages=['russian', 'english']).tokenize(
        text) for text in lst]


def tokenizer(text):
    '''
    Принимает на вход список предложений
    Возвращает массив (список списков) очищенных данных - без пунктаации и в нижнем регистре
    '''
    return Tokenizer(separator_type='BySense', token_types=['Word'], languages=['russian', 'english']).tokenize(
        text)


def sans_stops(lst):
    return [[word.lower() for word in text if word.lower() not in stop_words] for text in lst]


print("Loading data...")
rawdata = pd.read_csv('outA.csv', engine='python', delimiter=';', names=['Page', 'Date', 'Text'], parse_dates=['Date'])
rawdata.drop_duplicates(['Text'], inplace=True)

print("Tokenizing and removing stopwords...")
rawdata["Sans_stops"] = sans_stops(tokenize_list(rawdata.Text.to_list()))
print("Extracting terms...")
rawdata["Extracted terms"] = [TermExtractor()(" ".join(text), strings=True)[:5] for text in
                              rawdata["Sans_stops"].to_list()]

extracted_terms = rawdata["Extracted terms"].to_list()
extracted_terms = [" ".join(text) for text in extracted_terms]
print(extracted_terms)
# flat_extracted_terms = [item for sublist in extracted_terms for item in sublist]
# print(flat_extracted_terms)

# Проведём векторизацию
vectorizer = TfidfVectorizer(min_df=1, tokenizer=tokenizer, stop_words=stop_words, decode_error='ignore',
                             ngram_range=(1, 3), norm='l2')
# Создадим матрицу векторов
print("Creating TF-IDF matrix...")
tfidf_matrix = vectorizer.fit_transform(extracted_terms)

print("Clustering...")
clustered = DBSCAN().fit(tfidf_matrix)
print(clustered.labels_)
