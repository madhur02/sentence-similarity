from nltk.util import ngrams
import nltk
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from itertools import permutations
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def generate_grams(sent,ngram_range=5):
    token = nltk.word_tokenize(sent)
    token = [w for w in token if not w in stop_words]
    ngram_range = 5
    unigram_list = []
    all_grams = []
    new_all_grams = []
    for i in range(1,ngram_range+1):
        if i == 1:
            unigrams= list(ngrams(token,i))
            unigram = list(map(lambda x: x[0],unigrams))
            unigram_list.append(unigram)
        else:
            grams = list(ngrams(token,i))
            new_gram = [list(permutations(tuple_gram)) for tuple_gram in grams]
            all_combination = ['-'.join(single_tup) for new_tuple in new_gram for single_tup in new_tuple]
            all_grams.append(all_combination)
            new_all_grams+= all_combination

    return unigram_list , new_all_grams

def tagged_ngram_list(main_ngram_list):

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),
                        tags=[str(i)]) for i, _d in enumerate(main_ngram_list)]
    return (tagged_data)

def train_doc2_vec(tagged_data):
    max_epochs = 100
    vec_size = 20
    alpha = 0.025
    model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm =1)
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")


def doc2vec_handler():
    main_unigram_list = []
    main_ngram_list   = []
    sent_list = [" The little Jerry is being chased by Tom in the big yard",
                        "The blue cat is catching the brown mouse in the forecourt"]

    for sent in sent_list:
        unigram_list,ngram_list = generate_grams(sent)
        main_unigram_list += unigram_list
        main_ngram_list +=(ngram_list)

    tagged_data = tagged_ngram_list(main_ngram_list)
    print (tagged_data)
    train_doc2_vec(tagged_data)

doc2vec_handler()
