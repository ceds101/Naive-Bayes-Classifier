import os
import mailparser
import pickle
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
        for email_file in os.listdir(self.dataset_dir)[:20]: #[:n] means n emails para mas mabilis lng (hindi lahat map-parse)
            with open(self.dataset_dir + email_file, encoding="utf8", errors="ignore") as email_fp:
                email = mailparser.parse_from_file_obj(email_fp)
                email_row = DataFrame({"message":[email.body], 
                                        "class":[email_file.split(".")[-1]]})
                self.email_array = self.email_array.append(email_row, ignore_index=True)
                #print(email_file, "appended into array")

        #TRAIN EMAIL DATASET
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

        print(self.p_ham)
        print(self.p_spam)

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

        #Save probabilites and vocabulary to file
        self.save_p_words()

    def classify_email(self, email):
        pass

    def cross_validation(self):
        pass



    # FILE MANAGEMENT
    def save_p_words(self):
        with open('proby_words.dat', 'wb') as pwords_fp:
            pickle.dump(self.p_words, pwords_fp, pickle.HIGHEST_PROTOCOL)
    
    def load_p_words(self):
        with open('proby_words.dat', 'rb') as pwords_fp:
            self.p_words = pickle.load(pwords_fp)

    def clear_p_words(self):
        open("proby_words.dat", "w").close()
    
    # WORD COUNT MANAGEMENT
    # def save_word_counts(self):
    #     with open('word_counts.dat', 'wb') as wrdcnt_fp:
    #         pickle.dump(self.word_counts, wrdcnt_fp, pickle.HIGHEST_PROTOCOL)
    
    # def load_word_counts(self):
    #     with open('word_counts.dat', 'rb') as wrdcnt_fp:
    #         self.word_counts = pickle.load(wrdcnt_fp)

    # def clear_word_counts(self):
    #     open("word_counts.dat", "w").close()

    # VOCABULARY MANAGMENT
    # def save_vocabulary(self):
    #     with open('vocabulary.dat', 'wb') as vocab_fp:
    #         pickle.dump(self.vocabulary, vocab_fp, pickle.HIGHEST_PROTOCOL)
    
    # def load_vocabulary(self):
    #     with open('vocabulary.dat', 'rb') as vocab_fp:
    #         self.vocabulary = pickle.load(vocab_fp)

    # def clear_vocabulary(self):
    #     open("vocabulary.dat", "w").close()