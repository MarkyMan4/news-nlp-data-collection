# update all existing records in the database to get the publisher
# since the way I get the publisher has changed over time, this will make sure I'm getting the right publisher for all articles
# this doesn't actually run the updates, just generate update statements

import pandas as pd
import re

df = pd.read_csv('data/dev_articles.csv')
update_stmts = []

for i in range(len(df)):
    url = df.iloc[i]['url']

    match = re.search('//([a-z0-9-]+)\.([a-z0-9-]+)', url)

    if not match:
        print(url)

    publisher = match.group(2)

    if publisher in ['com', 'net', 'edu', 'org', 'gov', 'mil'] and match.group(1) != 'www': # covers most extensions I see
        publisher = match.group(1)

    update_stmts.append(f"update news_article set publisher = '{publisher}' where id = {df.iloc[i]['id']};\n")

with open('testing/db/update_publisher.sql', 'w') as f:
    f.writelines(update_stmts)