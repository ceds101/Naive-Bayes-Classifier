import os
import mailparser
import re
import math
import json
import time
import multiprocessing
import pandas as pd
import numpy as np

from functools import partial
from pandas import read_json, read_csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class EmailClassifier:
    #Parameters
    dataset_dir = None
    laplace_smoothing = 0
    preprocessor = None

    #Initialize Variables
    email_array = DataFrame({'message': [], 'class': []})
    email_train = DataFrame({'message': [], 'class': []})
    email_test = DataFrame({'message': [], 'class': []})

    vocabulary_len = None
    vocab_map = None
    word_counts = None
    email_classes = None

    p = {}
    n = {}

    p['ham'] = 0.0
    p['spam'] = 0.0
    n['ham'] = 0
    n['spam'] = 0

    def __init__(self, dataset_dir = None, laplace_smoothing = 0, preprocessor = None):
        self.preprocessor = preprocessor
        self.dataset_dir = dataset_dir
        self.laplace_smoothing = laplace_smoothing

    def calc_word_proby(self, word):
        # print('Proby for word:', word)

        p_word = {}
        p_word['ham'] = 0.0
        p_word['spam'] = 0.0

        #if current word is not on vocabulary use lambda/(n_class + |V|)
        if word not in self.vocab_map:
            p_word['ham'] = (self.laplace_smoothing)/(self.n['ham'] + self.vocabulary_len)
            p_word['spam'] = (self.laplace_smoothing)/(self.n['spam'] + self.vocabulary_len)
            # print('Done for word:', word)
            # print('Probability ham:', p_word['ham'])
            # print('Probability spam:', p_word['spam'])
            return p_word

        n_word = {}
        n_word['ham'] = 0
        n_word['spam'] = 0

        #iterate email_classes list
            #n_word[email_class] = n_word[class] + self.word_counts[index, vocab_map[word]]
        for index, curr_email_class in enumerate(self.email_train['class'].values):
            n_word[curr_email_class] = n_word[curr_email_class] + self.word_counts[index, self.vocab_map[word]]

        p_word['ham'] = (n_word['ham'] + self.laplace_smoothing)/(self.n['ham'] + self.vocabulary_len)
        p_word['spam'] = (n_word['spam'] + self.laplace_smoothing)/(self.n['spam'] + self.vocabulary_len)

        # print('Done for word:', word)
        # print('Probability ham:', p_word['ham'])
        # print('Probability spam:', p_word['spam'])
        return p_word

    def train(self):
        #PARSE EMAIL DATASET
        print('====READING EMAILS CSV====')
        start_parse = time.time()

        self.email_array = read_csv("dataset/trec07p_clean/emails.csv", index_col=False, nrows=100, encoding="utf-8")
        self.email_array = self.email_array.dropna()

        end_parse = time.time()
        print('====READING FINISHED====')
        print(end_parse - start_parse, "Seconds")

        print('====CLEANING DATASET=====')
        start_clean = time.time()
        self.email_array['message'] = self.email_array['message'].map(self.preprocessor.preprocess_text)
        self.email_array = self.email_array.dropna()
        end_clean = time.time()
        print(self.email_array)
        print('====CLEANING FINISHED=====')
        print(end_clean - start_clean, "Seconds")

        #SPLIT TRAINING SET AND TEST SET
        print('====SPLITTING TRAIN-TEST====')
        self.email_train, self.email_test = train_test_split(self.email_array, test_size=0.30, random_state=42)
        print('====DONE SPLITTING TRAIN-TEST===')

        #TRAIN EMAIL DATASET
        print('====TRAINING START====')
        start_train = time.time()
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english')

        # This will generate a matrix m x n (m: email instance, n: word in the vocabulary)
        # Each entry in the matrix represents how many times a word n appeared in a particular email instance m
        self.word_counts = vectorizer.fit_transform(self.email_train['message'].values) 
        vocabulary = vectorizer.get_feature_names()
        self.vocabulary_len = len(vocabulary)
        self.vocab_map = vectorizer.vocabulary_
        
        # Calculate the prior probabilites [p(ham), p(spam)]
        email_count = len(self.email_train.index.values)
        ham_spam_count = self.email_train['class'].value_counts()
        ham_count = ham_spam_count['ham']
        spam_count = ham_spam_count['spam']
        self.p['ham'] = ham_count/email_count
        self.p['spam'] = spam_count/email_count

        # Find n_spam and n_ham
        num_of_words_array = self.word_counts.sum(axis=1)
        for index, curr_email_class in enumerate(self.email_train['class'].values):
            self.n[curr_email_class] = self.n[curr_email_class] + num_of_words_array[index,0]
        
        end_train = time.time()
        print('====TRAINING END====')
        print(end_train - start_train, "Seconds")

        print("Train size:", email_count)
        print("Test size:", len(self.email_test.index))
        print(num_of_words_array.sum())

        #Save classifier attributes
        #self.save_classifier()

    def classify_email(self, email_string):
        email_class = None
        email_words = re.findall(r'\w+', email_string.lower()) #REPLACE WITH PREPROCESSOR
        
        v_ham = math.log(self.p['ham'])
        v_spam = math.log(self.p['spam'])

        # Calculate the likelihood probabilities of each word in the current email in each class 
        # [p(word_1|ham), p(word_1|spam), p(word_2|spam) ....]

        for word in email_words:
            curr_p_word = self.calc_word_proby(word)
            v_ham = v_ham + math.log(curr_p_word['ham'])
            v_spam = v_spam + math.log(curr_p_word['spam'])
        
        if v_ham > v_spam:
            email_class = "ham"
        else:
            email_class = "spam"
        
        print("Ham log likelihood: ", v_ham)
        print("Spam log likelihood: ", v_spam)
        
        return email_class

    def check_performance(self):
        email_actual_class = self.email_test['class'].values
        email_predicted_class = self.email_test['message'].map(self.classify_email)

        print(confusion_matrix(email_actual_class, email_predicted_class, labels=["ham", "spam"]))
        print("Accuracy:" , accuracy_score(email_actual_class, email_predicted_class))
        print(classification_report(email_actual_class, email_predicted_class, labels=["ham", "spam"], digits = 4))

    #Save classifier attributes to JSON file
    def save_classifier(self):
        classifier_attrs = {"dataset_dir": self.dataset_dir
                            ,"laplace_smoothing": self.laplace_smoothing
                            ,"vocabulary": self.vocabulary
                            ,"p_ham": self.p_ham
                            ,"p_spam": self.p_spam
                            ,"n_ham": int(self.n['ham'])
                            ,"n_spam": int(self.n['spam'])
                            ,"p_words": self.p_words
                            ,"email_test": self.email_test.to_json(orient='index')
                            }
        with open('classifier.json', 'w') as classifier_fp:
            json.dump(classifier_attrs, classifier_fp)
    
    def load_classifier(self):
        with open('classifier.json', 'r') as classifier_fp:
            classifier_attrs = json.load(classifier_fp)
            self.dataset_dir = classifier_attrs['dataset_dir']
            self.laplace_smoothing = classifier_attrs['laplace_smoothing']
            self.vocabulary = classifier_attrs['vocabulary']
            self.p_ham = classifier_attrs['p_ham']
            self.p_spam = classifier_attrs['p_spam']
            self.n['ham'] = classifier_attrs['n_ham']
            self.n['spam'] = classifier_attrs['n_spam']
            self.p_words = classifier_attrs['p_words']
            self.email_test = read_json(classifier_attrs['email_test'], orient='index')