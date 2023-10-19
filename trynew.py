import math

import pandas as pd

import numpy as np

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # example from https://www.kaggle.com/code/sainijagjit/text-classification-using-svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer

from sklearn.pipeline import Pipeline



def main():

    stop_list = stopwords.words('english')
    words_to_remove = ["but", "no", "nor", "not", "aren't", "couldn't",
                        "can't", "didn't", "doesn't", "hadn't", "hasn't", "haven't",
                        "isn't", "shouldn't", "wasn't", "weren't", "wouldn't", "won't"]
    
    #stop_list = text.ENGLISH_STOP_WORDS
    #words_to_remove = ["noone", "nothing", "couldnt", "hasnt", "not", "no",
    #                           "nobody", "nor", "cant", "never", "however", "but", "cannot"]

    # Remove specified words from the stop_list
    #stop_list = [word for word in stop_list if word not in words_to_remove]

    data = pd.read_table('./train.txt', names = ['label', 'review'])


    # Tokenize reviews
    data['tokens'] = data.apply(lambda x: nltk.word_tokenize(x['review']), axis = 1)

    # Remove stop words
    #data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop_list])

    # Apply Porter stemming
    stemmer = PorterStemmer()
    data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

    # Unify the strings once again (detokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(use_idf=True)),
        ('svm', SVC())
    ])

    for i in range(5):
        x_train, x_dev, y_train, y_dev = train_test_split(
            data['tokens'], 
            data['label'], 
            test_size=0.1
        )


        # Define the parameter grid to search -> VERIFY OTHER PARAMETERS
        param_grid = {
            'tfidf__max_features': [1000, 2000, 5000],  # Example TfidfVectorizer parameter
            'svm__C': [0.1, 1, 10],  # Example SVM parameter
            'svm__kernel': ['linear', 'rbf']
        }


        # Create a custom accuracy scorer for GridSearchCV
        scorer = make_scorer(accuracy_score)

        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer, verbose=1, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Get the best model and its parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_


        # Print the best parameters
        print("Best Parameters:", best_params)
        print("Grid Best Score:", grid_search.best_score_)

        # Make predictions on the validation set
        dev_predict = best_model.predict(x_dev)

        # Calculate accuracy and confusion matrix
        ba_train = accuracy_score(y_dev, dev_predict)
        cm_train = confusion_matrix(y_dev, dev_predict)
        
        print(f"Accuracy: {ba_train}\nConfusion Matrix:\n {cm_train}\n")

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# 87.8

if __name__ == "__main__":
    main()
