"""
    This script is for doing data collection and storing in news_article.
    It delegates the NLP work to the compute server since the VM does not
    have enough RAM to handle the machine learning portion.

    Steps:
        1. collect articles from reddit
        2. insert articles into news_article
        3. find which articles need to have an entry in news_articlenlp
            3.1. save these articles as a csv
            3.2. copy csv to the compute server
        4. invoke script on compute server to perform nlp
        5. retrieve results of nlp script from compute 
        6. store nlp results in news_articlenlp
"""

import os
import re
import praw
import json
import newspaper
from newspaper import Article
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from remote import RemoteConnection
from nlp import perform_nlp


logfile = open(f'logs/{datetime.now().strftime("%Y-%m-%d %H:%M")}', 'w') # save outputs so I can debug if needed
logfile.write('starting\n')

# load the secrets for Reddit and the database
with open('secrets.json') as file:
    secrets = json.load(file)

    client_id = secrets['client_id']
    client_secret = secrets['client_secret']
    user_agent = secrets['user_agent']
    connection_string = secrets['connection_string'] # for connecting to postgresql db
    server = secrets['compute_server']
    compute_username = secrets['compute_username']
    compute_password = secrets['compute_password']
    remote_scp_path = secrets['remote_scp_path'] # path for copying and getting files from the server

# create connection for database
db = create_engine(connection_string)

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
    for i, submission in enumerate(reddit.subreddit('worldnews').hot(limit=20)):
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

            match = re.search('//([a-z0-9-]+)\.([a-z0-9-]+)', url)

            # the regex isn't handling all possible URLs at the moment, just skip them for now if it didn't match
            if not match:
                continue

            publisher = match.group(2)

            # some URLs don't have a 'www' or something in front of the site name
            # so this fixes that by just taking the item in the first group of the regex
            # the site name could also be something like 'www.gov.uk', so in this case I don't want 'www' to be the publisher.
            # if the first group is 'www', I'll still keep the second group even if it is net, gov, edu, etc.
            if publisher in ['com', 'net', 'edu', 'org', 'gov', 'mil', 'ca'] and match.group(1) != 'www': # covers most extensions I see
                publisher = match.group(1)

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

def main():
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
    except Exception as e:
        logfile.write('failed\n\n')
        logfile.write(e)

# driver code
start_time = datetime.now()
main()
logfile.write(f'Total run time: {datetime.now() - start_time}\n')
logfile.close()