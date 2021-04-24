"""
    This script is for doing data collection and NLP all in one script.
    This can be used if the server has enough memory to handle loading
    the LDA model.
"""

import re
import praw
import json
import newspaper
from newspaper import Article
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

logfile = open(f'logs/{datetime.now().strftime("%Y-%m-%d %H:%M")}', 'w') # save outputs so I can debug if needed
logfile.write('starting\n')

# load the secrets for Reddit and the database
with open('secrets.json') as file:
    secrets = json.load(file)

    client_id = secrets['client_id']
    client_secret = secrets['client_secret']
    user_agent = secrets['user_agent']
    connection_string = secrets['connection_string'] # for connecting to postgresql db
    # connection_string = secrets['aws_connection_string'] # for connection to MySQL db on AWS

# create connection for database
db = create_engine(connection_string)

# load the trained LDA model for topic modeling
model = LdaMulticore.load('models/news_lda_model')

def get_articles() -> dict:
    """
    Use the Reddit API to get recently posted news articles

    Returns:
        dict: Dictionary of news articles
    """
    reddit = praw.Reddit(client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent)

    posts = {}
    for i, submission in enumerate(reddit.subreddit('worldnews').hot(limit=10)):
        # skip over the first item since it's just a discussion thread on the subreddit
        if i > 0:
            post_title = submission.title
            url = submission.url
            score = submission.score
            sub_id = submission.id

            # VM runs out of memory when I do this, instead, just use a regex to get the
            # publisher from the url

            # paper = newspaper.build(url)
            # publisher = paper.brand

            match = re.search('//[a-z]+\.[a-z]+', url).group(0)
            publisher = match[match.index('.') + 1 : ]

            # some URLs don't have a 'www' or something in front of the site name
            # so this fixes that by just taking the string after '//' and before '.'
            if publisher in ['com', '.net', 'edu', 'org', 'gov', 'mil']: # covers most extensions I see
                publisher = match[2:match.index('.')]

            # put a try/except here because parsing the article will sometimes return a 403 error
            try: 
                article = Article(url)
                article.download()
                article.parse()

                headline = article.title
                date_published = article.publish_date

                if date_published:
                    date_published = date_published.strftime('%Y-%m-%d %H:%M:%S')

                content = str(article.text).replace('\n', ' ').replace('  ', ' ')

                posts.update({
                    sub_id: {
                        'post_title': post_title,
                        'url': url,
                        'score': score,
                        'publisher': publisher,
                        'headline': headline,
                        'date_published': date_published,
                        'content': content
                    }
                })

            except Exception as e:
                logfile.write(f'error with URL: {url}\n')
                logfile.write(f'{e}\n')

    logfile.write(f'retrieved {len(posts.keys())} articles\n')
    return posts

def convert_to_dataframe(posts) -> pd.DataFrame:
    """
    converts a dictionary of posts to a dataframe to be inserted into table
    
    Arguments:
        posts {dict} -- dictionary of posts from reddit

    Returns:
        pd.DataFrame: Posts represented as a dataframe
    """
    df = pd.DataFrame(data=posts).transpose().reset_index()
    df = df.rename(columns={'index': 'post_id'})

    # remove null values for content and headline to satisfy db constraints
    df = df[df['content'].str.len() > 0]
    df = df[df['headline'].str.len() > 0]

    # if the date_published is null, fill it in with the current date
    df['date_published'].fillna(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), inplace=True)

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records from the DataFrame that already exit in the database.

    Args:
        df (pd.DataFrame): articles collected

    Returns:
        pd.DataFrame: Articles that don't yet exist in news_article
    """
    # remove any records from the dataframe that are already in the database
    # ids = pd.read_sql('SELECT DISTINCT post_id FROM NAP.article', con=db) # old query for MySQL db
    ids = pd.read_sql('SELECT DISTINCT post_id FROM news_article', con=db)
    ids = ids['post_id'].tolist()

    df = df[~df['post_id'].isin(ids)].reset_index(drop=True)

    return df

def insert_into_news_article(df: pd.DataFrame):
    """
    inserts a dataframe into the news_article table
    
    Arguments:
        df {[type]} -- [description]
    """
    # df.to_sql('article', con=db, if_exists='append', index=False) # article was the table in MySQL
    df.to_sql('news_article', con=db, if_exists='append', index=False)
    logfile.write(f'inserted {len(df)} records into news_article\n')

def find_articles_not_in_news_articlenlp() -> pd.DataFrame:
    """
    Finds articles that do not have an entry in news_articlenlp.
    These are returned so that NLP can be performed on them and 
    they can be added to news_articlenlp.

    Returns:
        pd.DataFrame: Articles in news_article that aren't in news_articlenlp
    """
    articles_without_nlp = pd.read_sql(
        """
        select *
        from news_article
        where id not in (select article_id from news_articlenlp)
        """,
        con=db
    )

    return articles_without_nlp

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

def insert_into_news_articlenlp(df: pd.DataFrame):
    """
    Performs sentiment analysis to find sentiment and subjectivity.
    Uses trained model to identify topic of article.
    Inserts NLP data into the news_articlenlp table.

    Args:
        df ([type]): [description]
    """
    df.to_sql('news_articlenlp', con=db, if_exists='append', index=False)
    logfile.write(f'inserted {len(df)} records into news_articlenlp\n')

def main(event=None, context=None):
    try:
        posts = get_articles()
        df = convert_to_dataframe(posts)
        df = remove_duplicates(df)

        if not df.empty:
            insert_into_news_article(df)
        else:
            logfile.write('no data to insert into news_article\n')

        articles_for_nlp = find_articles_not_in_news_articlenlp()

        if not articles_for_nlp.empty:
            article_nlp = perform_nlp(articles_for_nlp)
            insert_into_news_articlenlp(article_nlp)
        else:
            logfile.write('no data to insert into news_articlenlp\n')

        logfile.write('success\n')
    except:
        logfile.write('failed\n')

# driver code
start_time = datetime.now()
main()
logfile.write(f'Total run time: {datetime.now() - start_time}\n')
logfile.close()