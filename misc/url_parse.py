# get the publisher from the URL

import re

urls = [
    'https://www.gov.uk/government/news/',
    'https://www.i24news.tv/en/news/',
    'https://apnews.com/article/',
    'https://news.yahoo.com/'
]

for url in urls:
    match = re.search('//([a-z0-9]+)\.([a-z0-9]+)', url)
    print(match.group(0))
    print(match.group(1))
    print(match.group(2))

    # the regex isn't handling all possible URLs at the moment, just skip them for now if it didn't match
    # if match:
    #     match = match.group(0)
    # else:
    #     continue

    # publisher = match[match.index('.') + 1 : ]

    # # some URLs don't have a 'www' or something in front of the site name
    # # so this fixes that by just taking the string after '//' and before '.'
    # if publisher in ['com', 'net', 'edu', 'org', 'gov', 'mil']: # covers most extensions I see
    #     publisher = match[2:match.index('.')]

    # print(f'URL: {url}')
    # print(f'Publisher: {publisher}')
    print('------------------------------------')