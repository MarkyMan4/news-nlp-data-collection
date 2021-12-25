# get the publisher from the URL

import re

urls = [
    'https://www.gov.uk/government/news/',
    'https://www.i24news.tv/en/news/',
    'https://apnews.com/article/',
    'https://news.yahoo.com/',
    'https://thinkpol.ca/2021/08/11/'
]

for url in urls:
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

    print(url)
    print(publisher)
    print('-------------------------')
