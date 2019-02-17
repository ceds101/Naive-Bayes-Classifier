import utils
import classifiers.email_classifier

def main():
    email_classifier = classifiers.email_classifier.EmailClassifier('dataset/trec07p_clean/data/', 1)
    email_classifier.train()
    #email_classifier.load_vocabulary()
    print(sum(email_classifier.p_words['spam'].values()))

if __name__ == "__main__":
    main()