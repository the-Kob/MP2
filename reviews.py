import pandas as pd

import nltk
from nltk.stem.porter import *

from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import auxiliary

def read_test_data(file_name):

    auxiliary.print_header("Getting test file and reading data...")

    # Make a data set from the corresponding testing file
    data = pd.read_table(file_name, names = ['review'])
    
    print("\n Data read:")
    print(data.head(5)) # print 5 first rows of data 
    print("...")
    
    return data

def improved_preprocess_data(data):

    auxiliary.print_header("Preprocessing data...")

    tokenizer = RegexpTokenizer(r'[\w]+') # sequences of alphanumeric characters and underscores (it does not account for punctuation)

    # Tokenize reviews
    data['tokens'] = data.apply(lambda x: tokenizer.tokenize(x['review']), axis = 1)

    # Apply Porter stemming
    stemmer = PorterStemmer()
    data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

    # Unify the strings once again (detokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    print("\n Preprocessed data:")
    print(data["tokens"].head(5)) # print 5 first rows of data 
    print("...")

    return data

def main():

    auxiliary.download_packages()

    # Read the training file and preprocess data
    train_data = auxiliary.read_train_data('./train.txt')
    processed_train_data = improved_preprocess_data(train_data)

    # Read the test file and preprocess data
    test_data = read_test_data('./test_just_reviews.txt')
    processed_test_data = improved_preprocess_data(test_data)


    # Setup best model (parameters were obtained through gridsearch -> see fine_tuning_SVM.py)
    best_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(use_idf=True, ngram_range=(1,2), max_df=0.2, binary=True, smooth_idf=False)),
        ('svm', SVC(kernel='linear'))
    ])

    # Train model (since this is the final test, we use the entirety of the training data)
    auxiliary.print_header("Running best model: fine-tuned Support Vector Machine...")

    # Evalute model with average accuracy score (cross validation)
    print("\n## Average Accuracy")
    scores = cross_val_score(best_pipeline, processed_train_data["tokens"], processed_train_data["label"], cv=5, scoring='accuracy')
    print("Average accuracy of %0.5f with a standard deviation of %0.5f." % (scores.mean(), scores.std()))


    print("\n## Training For Final Test...")
    best_pipeline.fit(processed_train_data["tokens"], processed_train_data["label"])

    # Run model for test_just_reviews.txt
    test_predict = best_pipeline.predict(processed_test_data[['tokens']].squeeze('columns')) # get only the tokens, not the reviews) # WILL NOT WORK

    # Write results to results.txt
    auxiliary.print_header("Writing test_just_reviews.txt predictions to results.txt...")
    output = open("results.txt", "w")
    for item in test_predict:
            output.write(item + "\n")
    output.close()

    print("Done!! Check results.txt :)\n")


if __name__ == "__main__":
    main()
