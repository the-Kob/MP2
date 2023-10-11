import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import argparse
import random
import os
import torch.nn as nn
import utils
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
sns.set() # use seaborn plotting style

from sklearn.feature_extraction.text import CountVectorizer

# NEED NLTK (python -m nltk.downloader stopwords && python -m nltk.downloader punkt)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Preprocessing inspired by the example techniques shown in: https://faun.pub/natural-language-processing-nlp-data-preprocessing-techniques-8d6c957e2259

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_data(file, test_file = False):

    _preprocessed_labels = []
    _preprocessed_reviews = []

    for line in file:
        if not test_file:
            label, review = line.strip().split('\t')
            
            # Clean labels and reviews to make analysis more accurate
            preprocessed_label = clean_text(label)
            preprocessed_review = clean_text(review)

        else:
            review = line.strip()
            # Clean review to make analysis more accurate
            preprocessed_review = clean_text(review)


        # Tokenize data to promote standardization
        preprocessed_review = word_tokenize(preprocessed_review)

        # Stop word removal and stemming to reduce important words to their root form (promote standardization)
        preprocessed_review = [stemmer.stem(word) for word in preprocessed_review if word not in stop_words]

        # Add preprocessed data to their respective arrays
        if not test_file:
            _preprocessed_labels.append(preprocessed_label)

        _preprocessed_reviews.append(preprocessed_review)

    return _preprocessed_labels, _preprocessed_reviews

def clean_text(text):
    text = re.sub('<[^>]*>', '', text) # remove HTML tags
    text = re.sub('\d+', '', text) # remove numbers
    text = re.sub('[^\w\s]', '', text) # remove punctuation
    text = text.lower() # convert text to lowercase

    return text

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

class MultinomialNaiveBayes(object):
    def __init__(self):
        self.model = self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    def train(self, _train_x, _train_y):
        # Training the model with the training data
        self.model.fit(_train_x, _train_y)
    
    def evaluate(self, _test_x, _test_y):
        # Predicting the test data categories
        predicted_categories = self.model.predict(_test_x)

        # plotting the confusion matrix
        mat = confusion_matrix(_test_y, predicted_categories)
        sns.heatmap(mat.T, square = True, annot=True, fmt = "d")
        plt.xlabel("true labels")
        plt.ylabel("predicted label")
        plt.show()
        print("Accuracy: {}".format(accuracy_score(_test_y, predicted_categories)))

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features)) #Does our weight already have bias? np.zeros((n_classes, n_features + 1 ))....

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """

        # Calculate highest score -> get predicted class
        y_hat = np.dot(self.W, x_i).argmax(axis=0)

        #If a mistake is committed, correct it
        if(y_hat != y_i):
            self.W[y_i, :] += x_i.T # Increase weight of gold class
            self.W[y_hat, :] -= x_i.T # Decrease weight of incorrect class

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b - I think this is correct

        # Calculate scores according to the model (n_classes x 1).
        scores = np.dot(self.W, x_i)[:,None]

        # One-hot vector with the gold label (n_classes x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Conditional probability of y, according to softmax (n_classes x 1).
        # y_probabilities = Softmax(scores) not working properly
        z = np.sum(np.exp(scores))
        y_probabilities = np.exp(scores) / z

        # Update weights with stochastic gradient descent
        self.W += learning_rate * (y_one_hot - y_probabilities) * x_i[None, :]

class MLP(object):
    # Q2.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().

    def __init__(self, n_classes, n_features, hidden_size):
        self.nClasses = n_classes

        mu, sigma = 0.1, 0.1

        # Initialize weight matrices with normal distribution N(mu, sigma^2)
        W1 = np.random.normal(mu, sigma, size = (hidden_size, n_features))
        W2 = np.random.normal(mu, sigma, size = (n_classes, hidden_size))
        
        # Initialize bias to zeroes vector
        b1 = np.zeros(hidden_size) # (hidden_size)
        b2 = np.zeros(n_classes) # (n_classes)

        self.weights = [W1, W2]
        self.biases = [b1, b2]

        self.nClasses= n_classes

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        predictedLabels = np.empty((X.shape[0]))

        for x in range(X.shape[0]):
            z1 = np.dot(self.weights[0], X[x]) + self.biases[0]
            h1 = np.maximum(0, z1) # relu activation
            
            z2 = np.dot(self.weights[1], h1) + self.biases[1]

            probs = np.empty((10))
            for i in range(10):
                z2 -= np.max(z2) # anti-overflow
                probs[i] = np.exp(z2)[i] / sum(np.exp(z2))

            predictedLabels[x] = np.argmax(probs)

        return predictedLabels
    
    def evaluate(self, X, y):
        predictedLabels = self.predict(X)
        acc = np.sum((y == predictedLabels)) / y.shape[0]

        return acc

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            self.update_weights(x_i, y_i, learning_rate)

    def update_weights(self, x, y, eta):
        z1 = np.dot(self.weights[0], x) + self.biases[0]
        h1 = np.maximum(0, z1) # relu activation

        z2 = np.dot(self.weights[1], h1) + self.biases[1]

        probs = np.empty((10))
        for i in range(10):
            z2 -= np.max(z2) # anti-overflow
            probs[i] = np.exp(z2)[i] / sum(np.exp(z2))

        gradZ2 = probs - self.getOneHot(y)

        gradW2 = np.reshape(gradZ2, (gradZ2.shape[0], 1)) * (np.matrix.getT(h1))
        gradB2 = gradZ2

        gradH1 = np.dot(np.matrix.getT(self.weights[1]), np.reshape(gradZ2, (gradZ2.shape[0], 1)))

        # Relu derivative
        z1ReluD = np.empty((200))

        for i in range(200):
            if z1[i] <= 0:
                z1ReluD[i] = 0
            else:
                z1ReluD[i] = 1

        gradZ1 = gradH1 * np.reshape(z1ReluD, (z1ReluD.shape[0], 1))
        gradW1 = gradZ1 * x
        gradB1 = np.reshape(gradZ1, (gradZ1.shape[0]))

        self.weights[0] = self.weights[0] - eta * gradW1     
        self.biases[0] = self.biases[0] - eta * gradB1    
        self.weights[1] = self.weights[1] - eta * gradW2    
        self.biases[1] = self.biases[1] - eta * gradB2     

    def getOneHot(self, y):
        oneHot = np.zeros(self.nClasses)

        for i in range(self.nClasses):
            if i == y:
                oneHot[i] = 1
        
        return oneHot

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()

def main():

    # PREPROCESS DATA ================================================================

    train_file = './train.txt'
    test_file = './test_just_reviews.txt'

    preprocessed_labels = []
    preprocessed_reviews = []
    preprocessed_reviews_test = []

    try:
        # Open the train file for reading
        with open(train_file, 'r') as file:
            preprocessed_labels, preprocessed_reviews = preprocess_data(file)

    except FileNotFoundError:
        print(f'Error: File {train_file} not found.')

    try:
        # Open the test file for reading
        with open(test_file, 'r') as file:
            __ , preprocessed_reviews_test = preprocess_data(file, True)

    except FileNotFoundError:
        print(f'Error: File {test_file} not found.')

    #print(preprocessed_reviews[0])
    #print(preprocessed_labels[0])
    #print(preprocessed_reviews_test[0])

    # Split dataset into train and dev, convert them into np.arrays
    train_to_dev_index = int(len(preprocessed_reviews) * 0.9) # developer set is 10% of train set
    train_X, dev_X = np.array(preprocessed_reviews[:train_to_dev_index], dtype=object), np.array(preprocessed_reviews[train_to_dev_index:], dtype=object)
    train_Y, dev_Y = np.array(preprocessed_labels[:train_to_dev_index], dtype=object), np.array(preprocessed_labels[train_to_dev_index:], dtype=object)
    test_X = np.array(preprocessed_reviews_test, dtype=object)

    #print(dev_X[0]) # should be 1400 * 0.9 = 1261 (verified)
    #print(dev_Y[0]) # should be DECEPTIVENEGATIVE (verified)

    print(train_X[0])

    # =================================================================================

    # MODEL SETUPS ====================================================================

    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['naive_bayes', 'perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)



    add_bias = opt.model != "mlp"

    n_classes = np.unique(train_Y).size  # 4 (verified)
    n_feats = train_X.shape[0] # 1260 (verified) ??????????????????????'

    #print(n_classes)
    #print(n_feats)

    # initialize the model
    if opt.model == 'naive_bayes':
        model = MultinomialNaiveBayes()
    elif opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    elif opt.model == 'mlp':
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []

    if opt.model == 'naive_bayes':
        model.train(train_X, train_Y)
    else:
        for i in epochs:
            print('Training epoch {}'.format(i))
            train_order = np.random.permutation(train_X.shape[0])
            train_X = train_X[train_order]
            train_Y = train_Y[train_order]
            model.train_epoch(
                train_X,
                train_Y,
                learning_rate=opt.learning_rate
            )
            valid_accs.append(model.evaluate(dev_X, dev_Y))

    # =================================================================================

    # MAKE RESULTS FILE FOR TEST

    # Plot
    if opt.model == 'naive bayes':
        model.evaluate(dev_X, dev_Y)
    else:
        plot(epochs, valid_accs)

if __name__ == "__main__":
    main()

