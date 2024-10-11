import pandas as pd
import numpy as np
import pickle as pkl
import nltk
import time
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nltk.download('punkt_tab')
EMBEDDING_FILE = "w2v.pkl"

def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)

def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels

def get_tokens(inp_str):
    # Initialize NLTK tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK tokenizer not found, downloading...")
        nltk.download('punkt')
    return nltk.tokenize.word_tokenize(inp_str)

def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = None
    tfidf_train = None

    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)
    tfidf_train = vectorizer.fit_transform(training_documents)

    return vectorizer, tfidf_train

def w2v(word2vec, token):
    word_vector = np.zeros(300,)

    if token in word2vec:
        word_vector = word2vec[token]

    return word_vector


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


def instantiate_models():
    nb = GaussianNB()
    mlp = MLPClassifier(random_state=100)
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)


    return nb, logistic, svm, mlp

def train_model_tfidf(model, tfidf_train, training_labels):
    tfidf_t = tfidf_train.toarray()

    model.fit(tfidf_t, training_labels)

    return model

def train_model_w2v(model, word2vec, training_documents, training_labels):
    i = []
    for j in training_documents:
        i.append(string2vec(word2vec, j))
    
    i_arr = np.array(i)

    model.fit(i_arr, training_labels)

    return model


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

if __name__ == "__main__":
    print("*************** Loading data & processing *****************")
    print("Loading dataset.csv....")
    documents, labels = load_as_list("dataset.csv")
    
    print("Loading Word2Vec representations....")
    word2vec = load_w2v(EMBEDDING_FILE)

    print("Computing TFIDF representations....")
    vectorizer, tfidf_train = vectorize_train(documents)


    print("\n**************** Training models ***********************")
    print("Instantiating models....")
    nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models()
    nb_w2v, logistic_w2v, svm_w2v, mlp_w2v = instantiate_models()

    print("Training Naive Bayes models....")
    start = time.time() 
    nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    end = time.time()
    print("Naive Bayes + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    nb_w2v = train_model_w2v(nb_w2v, word2vec, documents, labels)
    end = time.time()
    print("Naive Bayes + w2v trained in {0} seconds".format(end - start))

    print("Training Logistic Regression models....")
    start = time.time()
    logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    end = time.time()
    print("Logistic Regression + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    logistic_w2v = train_model_w2v(logistic_w2v, word2vec, documents, labels)
    end = time.time()
    print("Logistic Regression + w2v trained in {0} seconds".format(end - start))

    print("Training SVM models....")
    start = time.time()
    svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)
    end = time.time()
    print("SVM + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    svm_w2v = train_model_w2v(svm_w2v, word2vec, documents, labels)
    end = time.time()
    print("SVM + w2v trained in {0} seconds".format(end - start))

    print("Training Multilayer Perceptron models....")
    start = time.time()
    mlp_tfidf = train_model_tfidf(mlp_tfidf, tfidf_train, labels)
    end = time.time()
    print("Multilayer Perceptron + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    mlp_w2v = train_model_w2v(mlp_w2v, word2vec, documents, labels)
    end = time.time()
    print("Multilayer Perceptron + w2v trained in {0} seconds".format(end - start))

    # print("Word2Vec embedding for {0}:\t{1}".format("vaccine", w2v(word2vec, "vaccine")))

    print("\n***************** Testing models ***************************")
    test_documents, test_labels = load_as_list("test.csv")  # Loading the dataset

    models_tfidf = [nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf]
    models_w2v = [nb_w2v, logistic_w2v, svm_w2v, mlp_w2v]
    model_names = ["Naive Bayes", "Logistic Regression", "SVM", "Multilayer Perceptron"]

    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row

    i = 0
    while i < len(models_tfidf): # Loop through models
        print("Making predictions for " + model_names[i] + "....")
        p, r, f, a = test_model_tfidf(models_tfidf[i], vectorizer, test_documents, test_labels)
        if models_tfidf[i] is None:  # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i] + " + TFIDF", "N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i] + " + TFIDF", p, r, f, a])

        p, r, f, a = test_model_w2v(models_w2v[i], word2vec, test_documents, test_labels)
        if models_w2v[i] is None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i]+" + w2v","N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i]+" + w2v", p, r, f, a])
        i += 1
    outfile.close()


    print("\n************** Beginning chatbot execution *******************")

    user_input = input("Welcome to the CS 421 chatbot!  What do you want to talk about today?\n")

    w2v_test = string2vec(word2vec, user_input) # This assumes you're using Word2Vec representations

    label = None

    if mlp_w2v is not None:
        label = mlp_w2v.predict(w2v_test.reshape(1, -1)) # This assumes you're using mlp_w2v; feel free to update

        if label == 0:
            print("Hmm, it seems like you're feeling a bit down.")
        elif label == 1:
            print("It sounds like you're in a positive mood!")
        else:
            print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
