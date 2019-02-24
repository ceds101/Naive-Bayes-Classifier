import utils
from classifiers.email_classifier import EmailClassifier
import mailparser

def main():
    email_classifier = EmailClassifier('dataset/trec07p_clean/data/', 1)
    #email_classifier.train()
    email_classifier.load_classifier()
    email_classifier.check_performance()
    #print(email_classifier.classify_email('buy viagra now'))
    #print(email_classifier.classify_email('hello there'))
if __name__ == "__main__":
    main()