import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar
from plotly.graph_objects import Treemap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #creating empty list to create dataframes.
    category_names_1 = []
    category_counts_1 = []
    category_names_2 = []
    category_counts_2 = []
    genre = []

    #get the counts for each categories
    for col in df.iloc[:,4:].columns:
        category_names_1.append(col)
        category_counts_1.append(df[col].sum())
        #get the percentages of categories in the genre.
        for value in df.genre.unique():
            data = df[df['genre'] == value]
            genre.append(value)
            category_names_2.append(col)
            category_counts_2.append(round(data[col].sum()/data.shape[0] * 100))


    #create a datframe for each category and their counts
    viz1 = pd.DataFrame({'Category' : category_names_1, 'Counts' : category_counts_1})
    #create a datframe for percentage of category in genre
    viz2 = pd.DataFrame({'Genre': genre,'Category' : category_names_2, 'Percentage' : category_counts_2})
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Treemap(
                    labels = list(viz1.Category.values),
                    parents = ['Categories'] * viz1.shape[0],
                    values = list(viz1.Counts.values)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
            }
        },
        {
            'data': [
                Treemap(
                    labels = list(viz2[viz2.Genre == 'direct'].Category.values),
                    parents = list(viz2[viz2.Genre == 'direct'].Genre.values),
                    values = list(viz2[viz2.Genre == 'direct'].Percentage.values)
                )],
                        'layout': {
                'title': 'Distribution of Message Categories in direct Genre',
            }
        },
        {
            'data': [
                Treemap(
                    labels = list(viz2[viz2.Genre == 'social'].Category.values),
                    parents = list(viz2[viz2.Genre == 'social'].Genre.values),
                    values = list(viz2[viz2.Genre == 'social'].Percentage.values)
                )],
                        'layout': {
                'title': 'Distribution of Message Categories in Social Genre',
            }
        },
        {
            'data': [
                Treemap(
                    labels = list(viz2[viz2.Genre == 'news'].Category.values),
                    parents = list(viz2[viz2.Genre == 'news'].Genre.values),
                    values = list(viz2[viz2.Genre == 'news'].Percentage.values)
                )],
                        'layout': {
                'title': 'Distribution of Message Categories in News Genre',
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()