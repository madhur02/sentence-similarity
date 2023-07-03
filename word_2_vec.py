import gensim
import pandas as pd
import os
from gensim.models import Word2Vec
import nltk
from nltk import word_tokenize
import pickle
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
corpus_path = r"C:\Users\G753903\Downloads\text8"

def train_word2_vec():
    content = open(corpus_path,'r').readlines()[0]
    words_token  = nltk.word_tokenize(content)
    filtered_token = [w for w in words_token if not w in stop_words]
    sent_length = 10
    total_length = len(filtered_token)
    (range1 , addition)= divmod(total_length,sent_length)
    new_sentence_list = []
    low = 0
    high = 10
    for i in range(0,range1):
        new_sentence_list.append(filtered_token[low:high])
        low += 10
        high +=10
    print ("Num of Sentences  -%s" %(range1))

    new_sentence_list += [["little","Jerry" ,"chased","Tom","big","yard"],["blue","cat","catch" ,"brown", "mouse" ,"forecourt"]]

    model = Word2Vec(new_sentence_list,window=7, min_count =3,size=20,sg=0)
    words = list(model.wv.vocab)

    print ("Num of Words  -%s" %(len(words)))

    print ("----------Load Word 2 Vec Unigram model ------")
    with open('word2_vec.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print ("Model Load Successfully....")

train_word2_vec()
