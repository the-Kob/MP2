import pandas as pd

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.feature_extraction import text
from sklearn.metrics import confusion_matrix, accuracy_score


def get_scores(y_real, predict):
    ba_train = accuracy_score(y_real, predict)
    cm_train = confusion_matrix(y_real, predict)

    return ba_train, cm_train 

def print_scores(train_scores, dev_scores):
    print("## Train Scores")
    print(f"Accuracy: {train_scores[0]}\nConfusion Matrix:\n {train_scores[1]}")
    print("\n\n## Dev Scores")
    print(f"Accuracy: {dev_scores[0]}\nConfusion Matrix:\n {dev_scores[1]}\n")

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def download_packages():
    print_header("Making sure the correct NLTK packages are installed...")

    nltk.download('punkt') # sentence tokenizer
    nltk.download('stopwords') # stopwords
    stopwords.words('english')

def define_stopwords(package=["nltk", "sklearn"], remove_negative_words=False):
    if package == "nltk":
        stop_list = stopwords.words('english') # uncomment for nltk stop words

        if(remove_negative_words):
            words_to_remove = ["but", "no", "nor", "not", "aren't", "couldn't",
                        "can't", "didn't", "doesn't", "hadn't", "hasn't", "haven't",
                        "isn't", "shouldn't", "wasn't", "weren't", "wouldn't", "won't"]

            # Remove specified words from the stop_list
            stop_list = [word for word in stop_list if word not in words_to_remove]

    elif(package == "sklearn"):
        stop_list = text.ENGLISH_STOP_WORDS # uncomment for sklearn stop words

        if(remove_negative_words):
            words_to_remove = ["noone", "nothing", "couldnt", "hasnt", "not", "no",
                               "nobody", "nor", "cant", "never", "however", "but", "cannot"]

            # Remove specified words from the stop_list
            stop_list = [word for word in stop_list if word not in words_to_remove]

    return stop_list


def read_train_data(file_name):

    print_header("Getting training file and reading data...")

    # Make a data set from the corresponding training file
    data = pd.read_table(file_name, names = ['label', 'review'])
    
    print("\n Data read:")
    print(data.head(5)) # print 5 first rows of data 
    print("...")
    
    return data

def write_sklearn_stop_words(filename):
    sklearn_stop_words = open(filename, "w")
    for item in text.ENGLISH_STOP_WORDS:
        sklearn_stop_words.write(item + "\n")
    sklearn_stop_words.close()

def write_nltk_stop_words(filename):
    nltk_stop_words = open(filename, "w")
    for item in stopwords.words('english'):
        nltk_stop_words.write(item + "\n")
    nltk_stop_words.close()

def eval_run_pipeline(pipeline, x_train, x_dev, y_train, y_dev):

    # Make predictions
    train_predict = pipeline.predict(x_train)
    dev_predict = pipeline.predict(x_dev)

    train_scores = get_scores(y_train, train_predict)
    dev_scores = get_scores(y_dev, dev_predict)

    print_scores(train_scores, dev_scores)

    
def eval_return_pipeline(pipeline, x_train, x_dev, y_train, y_dev):

    # Make predictions
    train_predict = pipeline.predict(x_train)
    dev_predict = pipeline.predict(x_dev)

    train_scores = get_scores(y_train, train_predict)
    dev_scores = get_scores(y_dev, dev_predict)

    print_scores(train_scores, dev_scores)

    return dev_predict


def get_incorrect_evaluations(y_dev, dev_predict, x_dev, data):
    
    print("\n## Incorrect Predictions")
    print("Instances where predictions were incorrect:")
    for i, (true_label, predicted_label) in enumerate(zip(y_dev, dev_predict)):
        
        if true_label != predicted_label:
            review_number = (data[data['tokens'] == x_dev.iloc[i]].index).values[0] + 1
            print(f"\nReview Number {review_number}: True Label: {true_label}, Predicted Label: {predicted_label}, \nText: {data['review'].iloc[review_number - 1]}")