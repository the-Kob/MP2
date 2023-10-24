import pandas as pd

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split

import auxiliary


def main():

    auxiliary.download_packages()

    data = pd.read_table('./train.txt', names = ['label', 'review'])

    tokenizer = RegexpTokenizer(r'[\w]+') # sequences of alphanumeric characters and underscores (it does not account for punctuation)

    # Tokenize reviews
    data['tokens'] = data.apply(lambda x: tokenizer.tokenize(x['review']), axis = 1)

    # Remove stop words
    #stop_list = auxiliary.define_stopwords("nltk", remove_negative_words=False)
    #data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop_list])

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

    auxiliary.print_header("Running fine-tuned model...")

    x_train, x_dev, y_train, y_dev = train_test_split(
        data['tokens'], 
        data['label'], 
        test_size=0.1
    )

    pipeline.fit(x_train, y_train)
    dev_predictions = auxiliary.eval_return_pipeline(pipeline, x_train, x_dev, y_train, y_dev)

    print("\n## Average Accuracy")
    scores = cross_val_score(pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("Fine tuned Support Vector Machine achieves an average accuracy of %0.5f with a standard deviation of %0.5f." % (scores.mean(), scores.std()))

    auxiliary.get_incorrect_evaluations(y_dev, dev_predictions, x_dev, data)
    print("\n")

    #For external analysis (like NLP-Telescope)
    #auxiliary.output_dev_gold_and_predicted(y_dev, dev_predictions, x_dev, data)

if __name__ == "__main__":
    main()
