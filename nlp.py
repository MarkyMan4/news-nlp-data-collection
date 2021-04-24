"""
    This should run on the compute server. The data collection script on the
    VM should scp a csv to compute, which will be picked up by this script to 
    perform the natural language processing tasks.
"""

import pandas as pd
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


model = LdaMulticore.load('NewsNLP/models/news_lda_model')

def preprocess(article: str) -> list:
    """
    Preprocess a news article for topic modeling.
    This involves removing stop words and any words less than
    three characters. It also word tokenizes sentences and 
    lemmatizes each word.

    Args:
        article (str): Article to preprocess

    Returns:
        list: list of lowercase, lemmatized tokens from the article
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(article.lower()) # make all articles lower case
    words = [] # words resulting from applying the filters

    for token in tokens:
        if len(token) > 3 and token not in stop_words:
            words.append(lemmatizer.lemmatize(token))
    
    return words

def predict_topic(article: str) -> int:
    """
    Use the trained LDA model to predict the topic of the given article.

    Args:
        preprocessed_article (str): article to predict the topic for

    Returns:
        int: predicted topic
    """
    preprocessed = preprocess(article)

    # create bag of words with preprocessed article
    bow = model.id2word.doc2bow(preprocessed)

    # make prediction
    pred = model[bow]

    # find the topic with the best match - predictions are given 
    # as a list of tuples with the form (topic_num, score)
    predicted_topic = pred[0][0]
    best_match = pred[0][1]

    for p in pred:
        if p[1] > best_match:
            predicted_topic = p[0]
            best_match = p[1]

    return predicted_topic

def perform_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Do sentiment analysis and topic modeling for the articles provided.

    Args:
        df (pd.DataFrame): Articles from news_article that aren't in news_articlenlp

    Returns:
        pd.DataFrame: Results of sentiment analysis. This dataframe should contain the columns
           sentiment, subjectviity, article_id and topic_id.          
    """
    results = {
        'sentiment': [],
        'subjectivity': [],
        'article_id': [],
        'topic': []
    }

    sentiments = []
    subjectivities = []

    for i in range(len(df)):
        results['article_id'].append(df.iloc[i]['id'])
        
        # get the polarity and subjectivity
        blob = TextBlob(df.iloc[i]['content'])
        results['sentiment'].append(blob.sentiment.polarity)
        results['subjectivity'].append(blob.sentiment.subjectivity)

        # get the predicted topic
        article_content = df.iloc[i]['content']
        topic = predict_topic(article_content)
        results['topic'].append(topic)

    article_nlp = pd.DataFrame(results)

    return article_nlp

# read in file that was copied to this server, then pass it to perform_nlp()
articles = pd.read_csv('NewsNLP/articles_for_nlp.csv')
nlp_result = perform_nlp(articles)
nlp_result.to_csv('NewsNLP/nlp_result.csv', index=False)
