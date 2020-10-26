import pandas as pd
from catboost.text_processing import Tokenizer
from gensim.models import FastText


def tokenizer(lst):
    '''
    Принимает на вход список предложений
    Возвращает массив (список списков) очищенных данных - без пунктаации и в нижнем регистре
    '''
    return [Tokenizer(separator_type='BySense', token_types=['Word'], languages=['russian', 'english']).tokenize(
        sentence.lower()) for sentence in lst]


rawdata = pd.read_csv('outA.csv', engine='python', delimiter=';', names=['Page', 'Date', 'Text'], parse_dates=['Date'])
rawdata.drop_duplicates(['Text'], inplace=True)

# Объединяем все записи для последующей токенизации и отказа от пунктуации
rawtext = rawdata['Text'].to_list()

# Разбиваем предложения на слова и удаляем пунктуацию
tokenized_sentences = tokenizer(rawtext)
print(tokenized_sentences)

# Создадим и обучим модель
model = FastText(size=300, window=3, min_count=3, workers=4, sg=1, negative=5, min_n=2, max_n=5, word_ngrams=1)
model.build_vocab(sentences=tokenized_sentences)
model.train(sentences=tokenized_sentences, total_examples=len(tokenized_sentences), epochs=1000)
