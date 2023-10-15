import math

import pandas as pd

import numpy as np

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # example from https://www.kaggle.com/code/sainijagjit/text-classification-using-svm
from sklearn.neighbors import KNeighborsClassifier # example from https://medium.com/@ashins1997/text-classification-456513e18893
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Preprocessing inspired by the example techniques shown in: https://anderfernandez.com/en/blog/naive-bayes-in-python/
# Naive Bayes model implementation inspired by: https://anderfernandez.com/en/blog/naive-bayes-in-python/


def get_scores(y_real, predict):
    ba_train = accuracy_score(y_real, predict)
    cm_train = confusion_matrix(y_real, predict)

    return ba_train, cm_train 

def print_scores(scores):
    return f"Balanced Accuracy: {scores[0]}\nConfussion Matrix:\n {scores[1]}"

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def download_packages():
    print_header("Making sure the correct NLTK packages are installed...")

    nltk.download('punkt') # sentence tokenizer
    nltk.download('stopwords') # stopwords
    stopwords.words('english')

def read_data(file_name, real_test=False):

    if(real_test):
        print_header("Getting testing file and reading data...")
    else:
        print_header("Getting training file and reading data...")

    # Make a data set from the corresponding training file
    if(real_test):
        data = pd.read_table(file_name, names = ['review'])
    else:
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

def data_preprocessing(data, stop_list):
    print_header("Preprocessing data...")

    # Tokenize reviews
    data['tokens'] = data.apply(lambda x: nltk.word_tokenize(x['review']), axis = 1)

    # Remove stop words
    data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop_list])

    # Apply Porter stemming
    stemmer = PorterStemmer() # CAN WE APPLY OTHER STEMMING PERHPS???
    data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

    # Unify the strings once again (detokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    return data

def split_train_dev_sets(data, percentage_dev=0.1):

    x_train, x_dev, y_train, y_dev = train_test_split(
        data['tokens'], 
        data['label'], 
        test_size= percentage_dev,
        random_state=None, # acts as a seed (shuffle will be the same each run)
        shuffle=False # eliminate this if you want to shuffle
    )
    
    return x_train, x_dev, y_train, y_dev

def create_vectorizer():

    vectorizer = CountVectorizer(
        strip_accents = 'ascii', 
        lowercase = True # CHANGE THIS PERHAPS?????????????????????????????????????
    )
    
    return vectorizer

def create_TF_matrix(x_train, x_dev):

    # Create vectorizer
    vectorizer = create_vectorizer()
    
    # Fit vectorizer & transform it
    vectorizer_fit = vectorizer.fit(x_train)
    x_train_transformed = vectorizer_fit.transform(x_train)
    x_dev_transformed = vectorizer_fit.transform(x_dev)

    return x_train_transformed, x_dev_transformed

def create_TF_matrix_real_test(data, x_train):

    vectorizer = create_vectorizer()
    
    # Fit vectorizer & transform it
    vectorizer_fit = vectorizer.fit(x_train)
    x_train_transformed = vectorizer_fit.transform(x_train)
    x_test_transformed = vectorizer_fit.transform(data)

    return x_train_transformed, x_test_transformed
    
def train_multinomial_NB(x_train_transformed, y_train, x_test_transformed):
    # Define model
    naive_bayes = MultinomialNB(alpha=3) # by default, smoothing is already 1... (alpha = 3 gets best values)
    naive_bayes.fit(x_train_transformed, y_train)

    # Make predictions
    train_predict = naive_bayes.predict(x_train_transformed)
    test_predict = naive_bayes.predict(x_test_transformed)

    return train_predict, test_predict

def train_logistic_regression(x_train_transformed, y_train, x_test_transformed):
    # Define model
    logistic_regression = LogisticRegression(max_iter=1400)
    logistic_regression.fit(x_train_transformed, y_train)

    # Make predictions
    train_predict = logistic_regression.predict(x_train_transformed)
    test_predict = logistic_regression.predict(x_test_transformed)

    return train_predict, test_predict

def train_svm(x_train_transformed, y_train, x_test_transformed):
    # Define model
    svm = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm.fit(x_train_transformed, y_train)

    # Make predictions
    train_predict = svm.predict(x_train_transformed)
    test_predict = svm.predict(x_test_transformed)

    return train_predict, test_predict

def train_knn(x_train_transformed, y_train, x_test_transformed):
    # Normalize data to prevent "data leakage"
    scaler = StandardScaler(with_mean=False)
    x_train_transformed = scaler.fit_transform(x_train_transformed)
    x_test_transformed = scaler.transform(x_test_transformed)

    # Find the best k value with cross validation
    k = find_best_k_value(x_train_transformed, y_train, x_test_transformed)

    # Define model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_transformed, y_train)

    # Make predictions
    train_predict = knn.predict(x_train_transformed)
    test_predict = knn.predict(x_test_transformed)

    return train_predict, test_predict

def find_best_k_value(x_train_transformed, y_train, x_test_transformed):
    # Create KNN Classifier
    knn = KNeighborsClassifier() 

    # Define a grid of k values to search for the best k
    param_grid = {'n_neighbors': np.arange(1, 31)}
    
    # Use GridSearchCV to find the best k using cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(x_train_transformed, y_train)
    
    best_k = grid_search.best_params_['n_neighbors']

    return best_k

def train_multiple_models(x_train_transformed, x_dev_transformed, y_train, y_dev):

    print_header("Running Multinomial Naive Bayes...")
    train_predict, dev_predict = train_multinomial_NB(x_train_transformed, y_train, x_dev_transformed)

    # Get scores (accuracy and confusion matrix)
    train_scores = get_scores(y_train, train_predict)
    dev_scores = get_scores(y_dev, dev_predict)


    print("## Train Scores")
    print(print_scores(train_scores))
    print("\n\n## Dev Scores")
    print(print_scores(dev_scores))

    # ELIMINATE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #print(x_train.iat[0])
    #print(y_train.iat[0])
    #print("\n")
    # ELIMINATE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # Train Logistic Regression
    print_header("Running Logistic Regression...")
    train_predict, dev_predict = train_logistic_regression(x_train_transformed, y_train, x_dev_transformed)

    # Get scores (accuracy and confusion matrix)
    train_scores = get_scores(y_train, train_predict)
    dev_scores = get_scores(y_dev, dev_predict)

    print("## Train Scores")
    print(print_scores(train_scores))
    print("\n\n## Dev Scores")
    print(print_scores(dev_scores))

    print("\n")

    # Train SVM
    print_header("Running SVM...")
    train_predict, dev_predict = train_svm(x_train_transformed, y_train, x_dev_transformed)

    # Get scores (accuracy and confusion matrix)
    train_scores = get_scores(y_train, train_predict)
    dev_scores = get_scores(y_dev, dev_predict)

    print("## Train Scores")
    print(print_scores(train_scores))
    print("\n\n## Dev Scores")
    print(print_scores(dev_scores))

    print("\n")

    # Train KNN
    print_header("Running KNN...")
    train_predict, dev_predict = train_knn(x_train_transformed, y_train, x_dev_transformed)

    # Get scores (accuracy and confusion matrix)
    train_scores = get_scores(y_train, train_predict)
    dev_scores = get_scores(y_dev, dev_predict)

    print("## Train Scores")
    print(print_scores(train_scores))
    print("\n\n## Dev Scores")
    print(print_scores(dev_scores))

    print("\n")


def best_model(x_train, y_train, x_test):
    train_predict, test_predict = train_multinomial_NB(x_train, y_train, x_test)
    return test_predict


def main():

    download_packages()

    answer = input("\nPerform real run (y or n): ") 
    real_run_condition = (answer == 'y' or answer == 'Y')

    # Used to verify stop words of each package
    # write_sklearn_stop_words("sklearn_stop_words.txt")
    # write_nltk_stop_words("nltk_stop_words.txt")

    stop_list = stopwords.words('english') # uncomment for nltk stop words
    #stop_list = text.ENGLISH_STOP_WORDS # uncomment for sklearn stop words

    # Train data preprocessing
    train_data = read_data('./train.txt')
    train_data = data_preprocessing(train_data, stop_list)

    x_train, x_dev, y_train, y_dev = split_train_dev_sets(train_data)

    if(real_run_condition):
        # Uncomment this if we want to use the full training set during the real run (don't think you're supposed to)
        #x_train = train_data['tokens']
        #y_train = train_data['label']

        # Test data preprocessing
        test_data = read_data('./test_just_reviews.txt', real_test=True)
        test_data = data_preprocessing(test_data, stop_list)

        test_data = test_data[['tokens']].squeeze('columns') # get only the tokens, not the reviews
        x_train_transformed, x_test_transformed = create_TF_matrix_real_test(test_data, x_train)
    else:
        x_train, x_dev, y_train, y_dev = split_train_dev_sets(train_data)
        x_train_transformed, x_dev_transformed = create_TF_matrix(x_train, x_dev)

    # TRAIN AND RUN MODELS ===========================================================
    if(real_run_condition):
        print_header("Running Best Model > Multinomial MB") # CHANGE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        y_test = best_model(x_train_transformed, y_train, x_test_transformed)

        # Write output file
        output = open("results.txt", "w")
        for item in y_test:
            output.write(item + "\n")
        output.close()

        print("Check the results.txt file! \n")

    else:
        train_multiple_models(x_train_transformed, x_dev_transformed, y_train, y_dev)


if __name__ == "__main__":
    main()

