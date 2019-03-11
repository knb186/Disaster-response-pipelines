import sys
import pandas as pd
import numpy as np
# from keras.utils import np_utils
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk

nltk.download('punkt')
nltk.download('wordnet')
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', con=engine)
    X = df['message']
    Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
            'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
            'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings',
            'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
            'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']]
    return X, Y, Y.columns


def tokenize(text):
    text = [item.lower() for item in text]

    tokens = [word_tokenize(item) for item in text]

    for tok in tokens:
        lemm = [WordNetLemmatizer().lemmatize(word) for word in tok]

    return lemm


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))])

    # parameters = {'vect__min_df': [0.1,0.5, 1],
    #              'clf__estimator__n_jobs':[1,2]}

    # cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    # model_filepath = 'ml_pipeline.sav'
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()