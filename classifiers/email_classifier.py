import os
import mailparser
import re
import pickle
import time
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

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
        for email_file in os.listdir(self.dataset_dir)[:100]: #[:n] means n emails para mas mabilis lng (hindi lahat map-parse)
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

    def classify_email(self, email_string):
        email_class = None
        p_new_word_ham = (self.laplace_smoothing)/(self.n_ham + len(self.vocabulary))
        p_new_word_spam = (self.laplace_smoothing)/(self.n_spam + len(self.vocabulary))

        email_words = re.findall(r'\w+', email_string.lower())
        
        v_ham = self.p_ham
        v_spam = self.p_spam
        for word in email_words:
            v_ham = v_ham * self.p_words['ham'].get(word, p_new_word_ham)
            v_spam = v_spam * self.p_words['spam'].get(word, p_new_word_spam)
        
        if v_ham > v_spam:
            email_class = "ham"
        else:
            email_class = "spam"
        
        print("Ham likelihood: ", v_ham)
        print("Spam likelihood: ", v_spam)
        
        return email_class

    def check_performance(self):
        pass
    
    #TEST (BROKEN) TODO: FIX
    def save_classifier(self):
        with open('classifier.dat', 'wb') as classifier_fp:
            pickle.dump(self.__dict__, classifier_fp, pickle.HIGHEST_PROTOCOL)
    
    def load_classifier(self):
        with open('classifier.dat', 'rb') as classifier_fp:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(classifier_fp))