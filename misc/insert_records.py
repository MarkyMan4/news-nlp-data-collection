# script to insert records into the news_article table
# this is a one-time script to get data transferred from the old db to the new one

import json
import pandas as pd
from sqlalchemy import create_engine

with open('secrets.json') as file:
    secrets = json.load(file)
    connection_string = secrets['connection_string']

db = create_engine(connection_string)
articles = pd.read_csv('articles.csv')
articles.to_sql('news_article', con=db, if_exists='append', index=False)
