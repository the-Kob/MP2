import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

import auxiliary

# Model training motivated by https://www.kaggle.com/code/sainijagjit/text-classification-using-svm and https://anderfernandez.com/en/blog/naive-bayes-in-python/
# Preprocessing inspired by the example techniques shown in: https://anderfernandez.com/en/blog/naive-bayes-in-python/


def data_preprocessing(data, stop_list):
    auxiliary.print_header("Preprocessing data...")

    # Tokenize reviews
    data['tokens'] = data.apply(lambda x: nltk.word_tokenize(x['review']), axis = 1)

    # Remove stop words
    data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop_list])

    # Apply Porter stemming
    stemmer = PorterStemmer()
    data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

    # Unify the strings once again (detokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    print("\n Preprocessed data:")
    print(data["tokens"].head(5)) # print 5 first rows of data 
    print("...")

    return data

def split_train_dev_sets(data, percentage_dev=0.1):
    
    x_train, x_dev, y_train, y_dev = train_test_split(
        data['tokens'], 
        data['label'], 
        test_size= percentage_dev
    )
    
    return x_train, x_dev, y_train, y_dev
    
def train_model(x_train, y_train, model=[MultinomialNB(), LogisticRegression(), SVC(), SGDClassifier()], vectorizer=[CountVectorizer(), TfidfVectorizer()]):

    # Define pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('model', model)
    ])

    pipeline.fit(x_train, y_train)

    return pipeline

def eval_run_pipeline(pipeline, x_train, x_dev, y_train, y_dev):

    # Make predictions
    train_predict = pipeline.predict(x_train)
    dev_predict = pipeline.predict(x_dev)

    train_scores = auxiliary.get_scores(y_train, train_predict)
    dev_scores = auxiliary.get_scores(y_dev, dev_predict)

    auxiliary.print_scores(train_scores, dev_scores)

def train_multiple_models(x_train, x_dev, y_train, y_dev, data):

    # MULTINOMIAL NAIVE BAYES ==========================================================================================================================
    auxiliary.print_header("Running Multinomial Naive Bayes...")
    current_pipeline = train_model(x_train, y_train, model=MultinomialNB(), vectorizer=TfidfVectorizer())

    # Get scores (accuracy and confusion matrix)
    eval_run_pipeline(current_pipeline, x_train, x_dev, y_train, y_dev)

    # Calculate cross validation (median) accuracy score 
    scores = cross_val_score(current_pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("Multinomial Naive Bayes achieves an average accuracy of %0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))
    # ==================================================================================================================================================


    # Logistic Regression ==============================================================================================================================
    auxiliary.print_header("Running Logistic Regression...")
    current_pipeline = train_model(x_train, y_train, model=LogisticRegression(), vectorizer=TfidfVectorizer())

    # Get scores (accuracy and confusion matrix)
    eval_run_pipeline(current_pipeline, x_train, x_dev, y_train, y_dev)

    # Calculate cross validation (median) accuracy score 
    scores = cross_val_score(current_pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("Logistic Regression achieves an average accuracy of %0.5f accuracy with a standard deviation of %0.5f." % (scores.mean(), scores.std()))
    # ==================================================================================================================================================


    # SVC ==============================================================================================================================================
    auxiliary.print_header("Running Support Vector Machine...")
    current_pipeline = train_model(x_train, y_train, model=SVC(), vectorizer=TfidfVectorizer())

    # Get scores (accuracy and confusion matrix)
    eval_run_pipeline(current_pipeline, x_train, x_dev, y_train, y_dev)

    # Calculate cross validation (median) accuracy score 
    scores = cross_val_score(current_pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("Support Vector Machine achieves an average accuracy of %0.5f accuracy with a standard deviation of %0.5f." % (scores.mean(), scores.std()))
    # ==================================================================================================================================================


    # SGC ==============================================================================================================================================
    auxiliary.print_header("Running Stochastic Gradient Descent...")
    current_pipeline = train_model(x_train, y_train, model=SGDClassifier(), vectorizer=TfidfVectorizer())

    # Get scores (accuracy and confusion matrix)
    eval_run_pipeline(current_pipeline, x_train, x_dev, y_train, y_dev)

    # Calculate cross validation (median) accuracy score 
    scores = cross_val_score(current_pipeline, data["tokens"], data["label"], cv=5, scoring='accuracy')
    print("Stochastic Gradient Descent achieves an average accuracy of %0.5f accuracy with a standard deviation of %0.5f." % (scores.mean(), scores.std()))
    # ==================================================================================================================================================

    print("\n=> The best model appears to be the Support Vector Machine...\n")

def main():

    auxiliary.download_packages()

    # Used to verify stop words of each package
    # auxiliary.write_sklearn_stop_words("sklearn_stop_words.txt")
    # auxiliary.write_nltk_stop_words("nltk_stop_words.txt")

    stop_list = auxiliary.define_stopwords("nltk", remove_negative_words=False)

    # Train data preprocessing
    train_data = auxiliary.read_train_data('./train.txt')
    train_data = data_preprocessing(train_data, stop_list)

    x_train, x_dev, y_train, y_dev = split_train_dev_sets(train_data)

    train_multiple_models(x_train, x_dev, y_train, y_dev, train_data)



if __name__ == "__main__":
    main()

