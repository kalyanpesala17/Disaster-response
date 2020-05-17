import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import pickle

from sqlalchemy import create_engine


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath,engine)
    X = df.message
    Y = df.iloc[:,4:]
    categories = Y.columns

    return X,Y,categories


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tokens.append(lemmatizer.lemmatize(tok).lower().strip())
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer = tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs = -1,n_estimators=10)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pipeline.predict(X_test)
    for i,col in enumerate(Y.columns):
        print(col,classification_report(y_test.iloc[:,i],y_pred[:,i]))

    print("Accuracy:", accuracy_score(y_test.values,y_pred))
    print("Precision:", precision_score(y_test.values,y_pred,average='samples'))
    print("Recall:", recall_score(y_test.values,y_pred,average='samples'))


def save_model(model, model_filepath):

    filename = 'model.sav'
    pickle.dump(cv, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()