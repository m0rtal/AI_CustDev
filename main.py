import pandas as pd
from catboost.text_processing import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

# nltk.download("stopwords")

stopwords = set(nltk.corpus.stopwords.words("russian"))


def tokenizer(str):
    '''
    Принимает на вход список предложений
    Возвращает массив (список списков) очищенных данных - без пунктаации и в нижнем регистре
    '''
    return Tokenizer(separator_type='BySense', token_types=['Word'], languages=['russian', 'english']).tokenize(
        str.lower())


print("Loading data...")
rawdata = pd.read_csv('outA.csv', engine='python', delimiter=';', names=['Page', 'Date', 'Text'], parse_dates=['Date'])
rawdata.drop_duplicates(['Text'], inplace=True)

# Объединяем все записи для последующей токенизации и отказа от пунктуации
rawtext = rawdata['Text'].to_list()

# Проведём векторизацию
print("Vectorizing...")
vectorizer = TfidfVectorizer(min_df=1, tokenizer=tokenizer, stop_words=stopwords, decode_error='ignore',
                             ngram_range=(1, 3), norm='l2')
# Создадим матрицу векторов
tfidf_matrix = vectorizer.fit_transform(rawtext)

# Разбиваем на кластеры
print("Clustering...")
classifier = KMeans(n_clusters=2, verbose=2, n_jobs=2)
clustered = classifier.fit(tfidf_matrix)

rawdata["Clusters"] = clustered.labels_
print(rawdata[["Text", "Clusters"]])
