from stanfordcorenlp import StanfordCoreNLP
import logging
import json
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

if __name__ == '__main__':
    sNLP = StanfordNLP()
    #text = 'A blog post using Stanford CoreNLP Server. Visit www.khalidalnajjar.com for more details.'
    #text = ' The little Jerry is being chased by Tom in the big yard.'
    #text = 'The blue cat is catching the brown mouse in the forecourt.'
    #text = "A blog post using Stanford CoreNLP Server."
    new_text = 'Drinking Water helps maintain the balance of Body Fluids.'
    data = sNLP.annotate(new_text)
    sentences = data['sentences']
    #print (sentences)
    for jj in sentences:
        print (jj)
        #print ('---'*100)
        #basic_dependent = (jj['basicDependencies'])
        print ('---'*100)
        print (basic_dependent)
        #tokens = sentences['tokens']
