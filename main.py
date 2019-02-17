import utils
from classifiers.email_classifier import EmailClassifier
import mailparser

def main():
    email_classifier = EmailClassifier('dataset/trec07p_clean/data/', 1)
    #email_classifier.train()
    email_classifier.load_classifier()


if __name__ == "__main__":
    main()