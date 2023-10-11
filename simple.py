import pandas as pd
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# NLTK
nltk.download('punkt')
nltk.download('stopwords')
stopwords.words('english')
# Sklearn
text.ENGLISH_STOP_WORDS


# Preprocessing inspired by the example techniques shown in: https://faun.pub/natural-language-processing-nlp-data-preprocessing-techniques-8d6c957e2259

def get_scores(y_real, predict):
    ba_train = balanced_accuracy_score(y_real, predict)
    cm_train = confusion_matrix(y_real, predict)

    return ba_train, cm_train 

def print_scores(scores):
    return f"Balanced Accuracy: {scores[0]}\nConfussion Matrix:\n {scores[1]}"


def main():

    train_file = './train.txt'

    data = pd.read_table(train_file,
                     names = ['type', 'message']
                     )
    
    print(data)

    stop = stopwords.words('english')

    # Tokenize
    data['tokens'] = data.apply(lambda x: nltk.word_tokenize(x['message']), axis = 1)

    # Remove stop words
    data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop])

    # Apply Porter stemming
    stemmer = PorterStemmer()
    data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

    # Unify the strings once again
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    # Make split
    x_train, x_test, y_train, y_test = train_test_split(
        data['tokens'], 
        data['type'], 
        test_size= 0.1,
        random_state=0, # acts as a seed (shuffle will be the same each run)
        shuffle=False # eliminate this if you want to shuffle
        )

    # Create vectorizer
    vectorizer = CountVectorizer(
        strip_accents = 'ascii', 
        lowercase = True
        )

    # Fit vectorizer & transform it
    vectorizer_fit = vectorizer.fit(x_train)
    x_train_transformed = vectorizer_fit.transform(x_train)
    x_test_transformed = vectorizer_fit.transform(x_test)

    # PREPROCESS DATA ================================================================

    # Train the model
    naive_bayes = MultinomialNB()
    naive_bayes_fit = naive_bayes.fit(x_train_transformed, y_train)

    # Make predictions
    train_predict = naive_bayes_fit.predict(x_train_transformed)
    test_predict = naive_bayes_fit.predict(x_test_transformed)

    train_scores = get_scores(y_train, train_predict)
    test_scores = get_scores(y_test, test_predict)


    print("## Train Scores")
    print(print_scores(train_scores))
    print("\n\n## Test Scores")
    print(print_scores(test_scores))

    print(x_train.iat[0])
    print(y_train.iat[0])
    print("\n")


    # LOGISTIC REGRESSION 

    logistic_regression = LogisticRegression(max_iter=1400)
    logistic_regression_fit = logistic_regression.fit(x_train_transformed, y_train)

    train_predict = logistic_regression_fit.predict(x_train_transformed)
    test_predict = logistic_regression_fit.predict(x_test_transformed)

    train_scores = get_scores(y_train, train_predict)
    test_scores = get_scores(y_test, test_predict)

    print("## Train Scores")
    print(print_scores(train_scores))
    print("\n\n## Test Scores")
    print(print_scores(test_scores))


if __name__ == "__main__":
    main()

