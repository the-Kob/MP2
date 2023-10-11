import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Preprocessing inspired by the example techniques shown in: https://faun.pub/natural-language-processing-nlp-data-preprocessing-techniques-8d6c957e2259

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Lists to store preprocessed data
preprocessed_labels = []
preprocessed_reviews = []

def main():
    train_file = './train.txt'

    try:
        # Open the input file for reading
        with open(train_file, 'r') as file:
            preprocess_data(file)

    except FileNotFoundError:
        print(f'Error: File {train_file} not found.')

def preprocess_data(file):
    for line in file:
        label, review = line.strip().split('\t')

        # Clean labels and reviews to make analysis more accurate
        preprocessed_label = clean_text(label)
        preprocessed_review = clean_text(review)

        # Tokenize data to promote standardization
        preprocessed_review = word_tokenize(preprocessed_review)

        # Stop word removal and stemming to reduce important words to their root form (promote standardization)
        preprocessed_review = [stemmer.stem(word) for word in preprocessed_review if word not in stop_words]

        # Add preprocessed data to their respective arrays
        preprocessed_labels.append(preprocessed_label)
        preprocessed_reviews.append(preprocessed_review)

def clean_text(text):
    text = re.sub('<[^>]*>', '', text) # remove HTML tags
    text = re.sub('\d+', '', text) # remove numbers
    text = re.sub('[^\w\s]', '', text) # remove punctuation
    text = text.lower() # convert text to lowercase

    return text

if __name__ == "__main__":
    main()