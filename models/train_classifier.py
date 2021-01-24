import sys

from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(
        'messages',
        con=engine
    )


    X = df['message']
    y = df.drop(['id', 'message', 'genre', 'original'], axis=1)
    
    category_names = y.columns
    
    return X, y, category_names



def tokenize(text):
        
    #remove punctuation
    text = re.sub(r"[,.;@#?!&$]+\ *", " ", text)

    #tokenize
    tokens = word_tokenize(text)
    
    
    #define the Lemmatizer
    lemmatizer = WordNetLemmatizer()

    #lemmatize and normalize to lowercase
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        #remove stop words
        if clean_tok not in set(stopwords.words('english')):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
   
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])
    
    
    parameters = {
        'vect__max_features':(None, 5000),
        'clf__estimator__n_estimators': [5, 10, 20]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1,verbose=2)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    print(f'accuracy = {accuracy.mean()} \n')
    
    preds = pd.DataFrame(y_pred, columns = category_names)
    
    for i, var in enumerate(preds.columns):
        print('Class: ',var)
        print(classification_report(y_true= y_test[:,i], y_pred = y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2)
        
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