import os
import mailparser
import re
import math
import json
import time
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

class EmailClassifier:
    #Parameters
    dataset_dir = None
    laplace_smoothing = 0

    #Initialize Variables
    email_array = DataFrame({'message': [], 'class': []})

    vocabulary = None
    word_counts = None

    p_ham = 0.0
    p_spam = 0.0
    n_ham = 0
    n_spam = 0

    p_words = {}
    p_words['ham'] = {}
    p_words['spam'] = {}

    def __init__(self, dataset_dir = None, laplace_smoothing = 0):
        self.dataset_dir = dataset_dir
        self.laplace_smoothing = laplace_smoothing
    
    def train(self):
        #PARSE EMAIL DATASET
        print('====PARSING EMAILS====')
        start_parse = time.time()
        for email_file in os.listdir(self.dataset_dir)[:300]: #[:n] means n emails para mas mabilis lng (hindi lahat map-parse)
            with open(self.dataset_dir + email_file, encoding="utf8", errors="ignore") as email_fp:
                email = mailparser.parse_from_file_obj(email_fp)
                email_row = DataFrame({"message":[email.body], 
                                        "class":[email_file.split(".")[-1]]})
                self.email_array = self.email_array.append(email_row, ignore_index=True)
                #print(email_file, "appended into array")
        end_parse = time.time()
        print('====PARSING FINISHED====')
        print(end_parse - start_parse, "Seconds")

        #TRAIN EMAIL DATASET
        print('====TRAINING START====')
        start_train = time.time()
        vectorizer = CountVectorizer()

        # This will generate a matrix m x n (m: email instance, n: word in the vocabulary)
        # Each entry in the matrix represents how many times a word n appeared in a particular email instance m
        self.word_counts = vectorizer.fit_transform(self.email_array['message'].values) 
        self.vocabulary = vectorizer.get_feature_names()

        # Calculate the prior probabilites [p(ham), p(spam)]
        email_count = len(self.email_array.index)
        ham_spam_count = self.email_array['class'].value_counts()
        ham_count = ham_spam_count['ham']
        spam_count = ham_spam_count['spam']
        self.p_ham = ham_count/email_count
        self.p_spam = spam_count/email_count

        # Find n_spam and n_ham
        num_of_words_array = self.word_counts.sum(axis=1)
        for email_index in range(email_count):
            if self.email_array.loc[email_index, 'class'] == "ham":
                self.n_ham = self.n_ham + num_of_words_array[email_index,0]
            else:
                self.n_spam = self.n_spam + num_of_words_array[email_index,0]
        
        # Calculate the likelihood probabilities of each word in the vocabulary in each class 
        # [p(word_1|ham), p(word_1|spam), p(word_2|spam) ....]
        vocab_map = vectorizer.vocabulary_

        for word in self.vocabulary:
            p_word_ham = 0.0
            p_word_spam = 0.0

            n_word_ham = 0
            n_word_spam = 0
            for email_index in range(email_count):
                if self.email_array.loc[email_index, 'class'] == "ham":
                    n_word_ham = n_word_ham + self.word_counts[email_index, vocab_map[word]]
                else:
                    n_word_spam = n_word_spam + self.word_counts[email_index, vocab_map[word]]

            p_word_ham = (n_word_ham + self.laplace_smoothing)/(self.n_ham + len(self.vocabulary))
            p_word_spam = (n_word_spam + self.laplace_smoothing)/(self.n_spam + len(self.vocabulary))

            self.p_words['ham'][word] = p_word_ham
            self.p_words['spam'][word] = p_word_spam
        
        end_train = time.time()
        print('====TRAINING END====')
        print(end_train - start_train, "Seconds")

        #Save classifier attributes
        self.save_classifier()

    #TODO: ewan ko if mali yung pag gamit nung laplace smoothing
    def classify_email(self, email_string):
        email_class = None
        p_new_word_ham = (self.laplace_smoothing)/(self.n_ham + len(self.vocabulary))
        p_new_word_spam = (self.laplace_smoothing)/(self.n_spam + len(self.vocabulary))

        email_words = re.findall(r'\w+', email_string.lower())
        
        v_ham = math.log(self.p_ham)
        v_spam = math.log(self.p_spam)
        for word in email_words:
            curr_p_word_ham = self.p_words['ham'].get(word, p_new_word_ham)
            curr_p_word_spam = self.p_words['spam'].get(word, p_new_word_spam)
            v_ham = v_ham + math.log(curr_p_word_ham)
            v_spam = v_spam + math.log(curr_p_word_spam)
        
        if v_ham > v_spam:
            email_class = "ham"
        else:
            email_class = "spam"
        
        print("Ham log likelihood: ", v_ham)
        print("Spam log likelihood: ", v_spam)
        
        return email_class

    #TODO: accuracy shits (stats),
    def check_performance(self):
        email_actual_class = []
        email_predicted_class = []

        for email_file in os.listdir(self.dataset_dir)[500:700]: #[start:end] means n emails para mas mabilis lng (hindi lahat map-parse)
            with open(self.dataset_dir + email_file, encoding="utf8", errors="ignore") as email_fp:
                email = mailparser.parse_from_file_obj(email_fp)

                email_actual_class.append(email_file.split(".")[-1])
                email_predicted_class.append(self.classify_email(email.body))

        print(confusion_matrix(email_actual_class, email_predicted_class, labels=["ham", "spam"]))
        print(classification_report(email_actual_class, email_predicted_class, labels=["ham", "spam"]))

    #Save classifier attributes to JSON file
    def save_classifier(self):
        classifier_attrs = {"dataset_dir": self.dataset_dir
                            ,"laplace_smoothing": self.laplace_smoothing
                            ,"vocabulary": self.vocabulary
                            ,"p_ham": self.p_ham
                            ,"p_spam": self.p_spam
                            ,"n_ham": int(self.n_ham)
                            ,"n_spam": int(self.n_spam)
                            ,"p_words": self.p_words
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
            self.n_ham = classifier_attrs['n_ham']
            self.n_spam = classifier_attrs['n_spam']
            self.p_words = classifier_attrs['p_words']