import re
import praw
import json
import newspaper
from newspaper import Article
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

logfile = open(f'logs/{datetime.now().strftime("%Y-%m-%d %H:%M")}', 'w') # save outputs so I can debug if needed

logfile.write('starting\n')

with open('secrets.json') as file:
    secrets = json.load(file)

    client_id = secrets['client_id']
    client_secret = secrets['client_secret']
    user_agent = secrets['user_agent']
    connection_string = secrets['connection_string'] # for connecting to postgresql db
    # connection_string = secrets['aws_connection_string'] # for connection to MySQL db on AWS

def get_articles():
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

def convert_to_dataframe(posts):
    """
    converts a dictionary of posts to a dataframe to be inserted into table
    
    Arguments:
        posts {dict} -- dictionary of posts from reddit
    """
    df = pd.DataFrame(data=posts).transpose().reset_index()
    df = df.rename(columns={'index': 'post_id'})

    # remove null values for content and headline to satisfy db constraints
    df = df[df['content'].str.len() > 0]
    df = df[df['headline'].str.len() > 0]

    # if the date_published is null, fill it in with the current date
    df['date_published'].fillna(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), inplace=True)

    return df

def insert_into_db(df):
    """
    inserts a dataframe into the article table
    
    Arguments:
        df {[type]} -- [description]
    """
    db = create_engine(connection_string)

    # remove any records from the dataframe that are already in the database
    # ids = pd.read_sql('SELECT DISTINCT post_id FROM NAP.article', con=db) # old query for MySQL db
    ids = pd.read_sql('SELECT DISTINCT post_id FROM news_article', con=db)
    ids = ids['post_id'].tolist()

    df = df[~df['post_id'].isin(ids)].reset_index(drop=True)

    # don't try to insert if the dataframe is empty
    if df.empty:
        logfile.write('no data to insert\n')
        return

    # df.to_sql('article', con=db, if_exists='append', index=False) # article was the table in MySQL
    df.to_sql('news_article', con=db, if_exists='append', index=False)
    logfile.write(f'inserted {len(df)} records\n')


def main(event=None, context=None):
    try:
        posts = get_articles()
        df = convert_to_dataframe(posts)
        insert_into_db(df)

        logfile.write('success\n')
    except:
        logfile.write('failed\n')

# driver code
start_time = datetime.now()
main()
logfile.write(f'Total run time: {datetime.now() - start_time}\n')
logfile.close()