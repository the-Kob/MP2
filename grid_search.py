import math

import pandas as pd

import numpy as np

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # example from https://www.kaggle.com/code/sainijagjit/text-classification-using-svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from nltk.tokenize import RegexpTokenizer



# Only take out "that", "and", "the"
# eliminate very symbol except "!" (and perhaps ?)
# max_df=0.25, use_dif = 0.25, C = 1


def main():

    #stop_list = stopwords.words('english')
    #words_to_remove = ["but", "no", "nor", "not", "aren't", "couldn't",
    #                    "can't", "didn't", "doesn't", "hadn't", "hasn't", "haven't",
    #                    "isn't", "shouldn't", "wasn't", "weren't", "wouldn't", "won't"]
    
    #stop_list = text.ENGLISH_STOP_WORDS
    #words_to_remove = ["noone", "nothing", "couldnt", "hasnt", "not", "no",
    #                           "nobody", "nor", "cant", "never", "however", "but", "cannot"]

    # Remove specified words from the stop_list
    #stop_list = [word for word in stop_list if word not in words_to_remove]

    

    data = pd.read_table('./train.txt', names = ['label', 'review'])

    tokenizer = RegexpTokenizer(r'[\w]+') # sequences of alphanumeric characters and underscores (it does not account for punctuation)

    # Tokenize reviews
    data['tokens'] = data.apply(lambda x: tokenizer.tokenize(x['review']), axis = 1)

    # Remove stop words
    #data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop_list])

    # Apply Porter stemming
    stemmer = PorterStemmer()
    data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

    # Unify the strings once again (detokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(use_idf=True, ngram_range=(1,2), max_df=0.2, binary=True, smooth_idf=False)),
        ('svm', SVC(kernel='linear'))
    ])


    #param_grid = {
    #'svm__break_ties': [True, False]
    #}

    # Perform Grid Search with cross-validation
    #grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
    #grid_search.fit(data["tokens"], data["label"])

    # Get the best model and its parameters
    #best_model = grid_search.best_estimator_
    #best_params = grid_search.best_params_


    # Print the best parameters
    #print("Best Parameters:", best_params)
    #print("Grid Best Score: %0.5f" % grid_search.best_score_)

    scores = cross_val_score(pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))

    #0.82929
    #0.84143
    #0.85214
    #0.85286

if __name__ == "__main__":
    main()
