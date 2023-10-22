import pandas as pd

import nltk
from nltk.stem.porter import *

from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import auxiliary


def main():

    auxiliary.download_packages()

    stop_list = auxiliary.define_stopwords("nltk", remove_negative_words=False)

    data = pd.read_table('./train.txt', names = ['label', 'review'])

    tokenizer = RegexpTokenizer(r'[\w]+') # sequences of alphanumeric characters and underscores (it does not account for punctuation)

    # Tokenize reviews
    data['tokens'] = data.apply(lambda x: tokenizer.tokenize(x['review']), axis = 1)

    # Apply Porter stemming
    stemmer = PorterStemmer()
    data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

    # Unify the strings once again (detokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    # These parameters were obtained through multiple executions of GridSearchCV, considering
    # the average accuracy score obtained from cross validation (seen below)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(use_idf=True, ngram_range=(1,2), max_df=0.2, binary=True, smooth_idf=False)),
        ('svm', SVC(kernel='linear'))
    ])

    auxiliary.print_header("Calculating average accuracy for the best model...")

    scores = cross_val_score(pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("Fine tuned Support Vector Machine achieves an average accuracy of %0.5f with a standard deviation of %0.5f.\n" % (scores.mean(), scores.std()))

if __name__ == "__main__":
    main()
