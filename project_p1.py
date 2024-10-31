import pandas as pd
import numpy as np
import pickle as pkl
import nltk
nltk.download('vader_lexicon')
import time
import csv
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer

EMBEDDING_FILE = "w2v.pkl"

def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)

def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels

def extract_user_info(user_input):
    name = ""
    name_match = re.search(r"(^|\s)([A-Z][A-Za-z-&'\.]*(\s|$)){2,4}", user_input)
    if name_match is not None:
        name = name_match.group(0).strip()
    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    # Initialize NLTK tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK tokenizer not found, downloading...")
        nltk.download('punkt')
    return nltk.tokenize.word_tokenize(inp_str)


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    vectorizer = None
    tfidf_train = None
    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)
    tfidf_train = vectorizer.fit_transform(training_documents)

    return vectorizer, tfidf_train


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300,)

    if token in word2vec:
        word_vector = word2vec[token]

    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300,)
    tokens_list = []
    extracted_tokens = get_tokens(user_input)

    for token in extracted_tokens:
        current_embedding = w2v(word2vec, token)
        tokens_list.append(current_embedding)

    if tokens_list:
        embedding = np.mean(tokens_list, axis=0)

    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Four instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    nb = GaussianNB()
    mlp = MLPClassifier(random_state=100)
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)

    return nb, logistic, svm, mlp


# Function: train_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using TFIDF
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    tfidf_t = tfidf_train.toarray()
    model.fit(tfidf_t, training_labels)

    return model


# Function: train_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_w2v(model, word2vec, training_documents, training_labels):
    i = []
    for j in training_documents:
        i.append(string2vec(word2vec, j))
    
    i_arr = np.array(i)

    model.fit(i_arr, training_labels)

    return model


# Function: test_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    tfidf_test = vectorizer.transform(test_documents)
    tfidf_test = tfidf_test.toarray()
    predicted_labels = model.predict(tfidf_test)

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy


# Function: test_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_w2v(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    test_embeddings = []
    for doc in test_documents:
        embedding = string2vec(word2vec, doc)
        test_embeddings.append(embedding)

    predicted_labels = model.predict(test_embeddings)

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy

# Function: compute_ttr(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the type-token ratio for tokens in the input string.
# Type-token ratio is computed as: num_types / num_tokens, where num_types is
# the number of unique tokens.
def compute_ttr(user_input):
    ttr = 0.0

    t = get_tokens(user_input)
    num_types = len(set(t))
    num_tokens = len(t)

    if num_tokens > 0:
        ttr = num_types / num_tokens
    else:
        ttr = 0

    return ttr


# Function: tokens_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of tokens per sentence
def tokens_per_sentence(user_input):
    tps = 0.0
    sentences = nltk.tokenize.sent_tokenize(user_input)
    total_tokens = 0
    for sentence in sentences:
        tokens = get_tokens(sentence)
        total_tokens += len(tokens)

    num_sentences = len(sentences)
    if num_sentences > 0:
        tps = total_tokens / num_sentences
    else:
        tps = 0

    return tps


# Function: get_dependency_parse(input)
# This function accepts a raw string input and returns a CoNLL-formatted output
# string with each line indicating a word, its POS tag, the index of its head
# word, and its relation to the head word.
# Parameters:
# input - A string containing a single text input (e.g., a sentence).
# Returns:
# output - A string containing one row per word, with each row containing the
#          word, its POS tag, the index of its head word, and its relation to
#          the head word.
def get_dependency_parse(input: str):
    output = ""
    dep_parser = CoreNLPDependencyParser(url="http://localhost:9000")

    parses = dep_parser.raw_parse(input)
    parse = next(parses)
    output = parse.to_conll(4)

    return output


# Function: get_dep_categories(parsed_input)
# parsed_input: A CONLL-formatted string.
# Returns: Five integers, corresponding to the number of nominal subjects (nsubj),
#          direct objects (obj), indirect objects (iobj), nominal modifiers (nmod),
#          and adjectival modifiers (amod) in the input, respectively.
#
# This function counts the number of grammatical relations belonging to each of five
# universal dependency relation categories specified for the provided input.
def get_dep_categories(parsed_input):
    num_nsubj = 0
    num_obj = 0
    num_iobj = 0
    num_nmod = 0
    num_amod = 0

    lines = parsed_input.strip().split('\n')
    for line in lines:
        if line: 
            parsed_line = line.split('\t')
            if len(parsed_line) >= 4:
                parse_relation = parsed_line[3]
                if parse_relation.startswith('nsubj'):
                    num_nsubj += 1
                elif parse_relation == 'obj':
                    num_obj += 1
                elif parse_relation == 'iobj':
                    num_iobj += 1
                elif parse_relation.startswith('nmod'):
                    num_nmod += 1
                elif parse_relation == 'amod':
                    num_amod += 1

    return num_nsubj, num_obj, num_iobj, num_nmod, num_amod


# Function: custom_feature_1(user_input)
# user_input: A string of arbitrary length
# Returns: An output specific to the feature type implemented.
#
# This function implements a custom stylistic feature extractor.
def custom_feature_1(user_input):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(user_input)
    
    intensity_score = abs(sentiment_scores['compound']) 
    
    # Here I analyze punctuation for urgency
    urgency_score = user_input.count('!') + user_input.count('?')
    
    # This part of the feature is to analyze modal verbs for urgency
    modal_verbs = ['must', 'should', 'need', 'ought', 'shall', 'will',  
    'can', 'could', 'may', 'might', 'have to', 'required', 'mandatory',
    'dare', 'used to', 'had better', 'ought to']
    words = get_tokens(user_input.lower())  
    modal_count = 0
    for word in words:
        if word in modal_verbs:
            modal_count += 1

    urgency_score += modal_count  
    
    # Here I analyze emotive language intensity
    emotive_words = ['love', 'hate', 'excited', 'afraid', 'surprised', 'angry',
        'joyful', 'sad', 'happy', 'disgusted', 'furious', 'delighted', 'terrified',
        'anxious', 'nervous', 'worried', 'elated', 'heartbroken', 'depressed', 'content',
        'pleased', 'upset', 'bitter', 'frightened', 'eager', 'hopeful', 'mournful']
    emotive_count = 0
    for word in words:
        if word in emotive_words:
            emotive_count += 1

    intensity_score += 0.1 * emotive_count
    
    # Detecting capitalization
    cap_match = re.findall(r'\b[A-Z]{2,}\b', user_input) 
    caps_count = len(cap_match)
    if caps_count > 0:
        intensity_score += 0.05 * caps_count
    
    # Adverb detection
    pos_tags = nltk.pos_tag(words)
    adverb_count = 0
    for word, tag in pos_tags:
        if tag.startswith('RB'):
            adverb_count += 1

    intensity_score += 0.05 * adverb_count

    return {'intensity_score': intensity_score, 'urgency_score': urgency_score}


# Function: custom_feature_2(user_input)
# user_input: A string of arbitrary length
# Returns: An output specific to the feature type implemented.
#
# This function implements a custom stylistic feature extractor.
def custom_feature_2(user_input):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(user_input)

    if sentiment_scores['compound'] >= 0.05:
        tone = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        tone = "Negative"
    else:
        tone = "Neutral"

    return tone


if __name__ == "__main__":

    # Load the dataset
    documents, labels = load_as_list("dataset.csv")


    vectorizer, tfidf_train = vectorize_train(documents)

    nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models() 
    svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)


    print("*********** Beginning chatbot execution *************************\n")

    # Display a welcome message to the user, and accept a user response of arbitrary length
    user_input = input("Welcome to the CS 421 chatbot!  What is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    # Query the user for a response
    user_input = input(f"Thanks {name}!  What do you want to talk about today?\n")

    # Predict user's sentiment
    tfidf_test = vectorizer.transform([user_input])  
    # w2v_test = string2vec(word2vec, user_input)  

    label = None
    label = svm_tfidf.predict(tfidf_test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    ttr = compute_ttr(user_input)
    tps = tokens_per_sentence(user_input)
    dep_parse = get_dependency_parse(user_input)
    num_nsubj, num_obj, num_iobj, num_nmod, num_amod = get_dep_categories(dep_parse)
    custom_1 = custom_feature_1(user_input)
    custom_2 = custom_feature_2(user_input)

    # Generate a stylistic analysis of the user's input
    print("Thanks!  Here's what I discovered about your writing style.")
    print("Type-Token Ratio: {0}".format(ttr))
    print("Average Tokens Per Sentence: {0}".format(tps))
    print("Dependencies:\n{0}".format(dep_parse)) # Uncomment to view the full dependency parse.
    print("# Nominal Subjects: {0}\n# Direct Objects: {1}\n# Indirect Objects: {2}"
          "\n# Nominal Modifiers: {3}\n# Adjectival Modifiers: {4}".format(num_nsubj, num_obj,
                                                                           num_iobj, num_nmod, num_amod))
    print("Custom Feature #1: {0}".format(custom_1))
    print("Custom Feature #2: {0}".format(custom_2))
    # ----------------------------------------------------------------------------
