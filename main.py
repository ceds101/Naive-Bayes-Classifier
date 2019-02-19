import utils
from classifiers.email_classifier import EmailClassifier
import mailparser

def main():
    email_classifier = EmailClassifier('dataset/trec07p_clean/data/', 1)
    email_classifier.load_classifier()
    #email_classifier.load_classifier()
    # with open('dataset/trec07p_clean/data/inmail.75417.ham', encoding="utf8", errors="ignore") as email_fp:
    #     email = mailparser.parse_from_file_obj(email_fp)
    #     print(email_classifier.classify_email(email.body))
    email_classifier.check_performance()

    #print(email_classifier.classify_email('academic research'))
if __name__ == "__main__":
    main()