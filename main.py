import utils
import classifiers.email_classifier

def main():
    email_classifier = classifiers.email_classifier.EmailClassifier('dataset/trec07p_clean/data/', 1)
    email_classifier.load_classifier()

    #print(email_classifier.p_words['spam'])
    print(email_classifier.classify_email('academic research'))

if __name__ == "__main__":
    main()