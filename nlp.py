"""
    This should run on the compute server. The data collection script on the
    VM should scp a csv to compute, which will be picked up by this script to 
    perform the natural language processing tasks.
"""

import math
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


model = LdaMulticore.load('models/news_lda_model')

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

def get_unique_terms(tokens: list) -> list:
    # find the unique terms, then count how many times each term appears
    unique_terms = []

    for token in tokens:
        if token not in unique_terms:
            unique_terms.append(token)

    return unique_terms

def get_term_frequency(unique_terms: list, all_tokens: list) -> dict:
    # find how many times each term appears
    term_counts = {}
    for term in unique_terms:
        term_counts.update({term: 0})

    for token in all_tokens:
        term_counts[token] += 1

    # find term frequencies by diving # of times each term appears by the term counts
    term_freqs = {}
    num_terms = len(all_tokens)

    for term in unique_terms:
        term_freqs.update({term: term_counts[term] / num_terms})

    return term_freqs

def get_inverse_document_frequency(content: str, unique_terms: list) -> dict:
    # split content into sentences
    sentences = sent_tokenize(content)
    num_sentences = len(sentences)

    # split each sentence into word tokens, no need to remove stop words here
    sentences = [word_tokenize(sent) for sent in sentences]

    # find number of sentences containing each term
    sentence_freqs = {}

    for term in unique_terms:
        sentence_freqs.update({term: 0})
        
    for term in unique_terms:
        for sent in sentences:
            if term in sent:
                sentence_freqs[term] += 1

    # compute inverse document frequency for each term
    idf = {}

    for term in unique_terms:
        term_val = 0

        # avoid division by 0
        if sentence_freqs[term] != 0:
            term_val = math.log(num_sentences / sentence_freqs[term])

        idf.update({
            term: term_val
        })

    return idf

def get_tf_idf(unique_terms: list, term_freqs: dict, idf: dict) -> dict:
    # find tfidf for each term
    tfidf = {}

    for term in unique_terms:
        tfidf.update({
            term: term_freqs[term] * idf[term]
        })

    return tfidf

def find_keywords(content: str) -> str:
    """
    Find the top ten key words using the TF-IDF calculation.

    Term Frequency = (# of times term appears) / (total # of terms in article)
    Inverse Document Frequency = log(# of sentences / # of sentences with the term)
    TF-IDF - term frequency * inverse document frequency

    Higher TF-IDF score means the term is more important.

    Args:
        content (str): content from news article to find keywords for

    Returns:
        str: semi-colon separated list of keywords
    """
    # tokenize the content and remove stopwords and punctuation
    sentences = sent_tokenize(content)
    tokens = []
    # sent tokenize first so the way it creates tokens is consistent with how it's done when computing IDF
    for sent in sentences:
        tokens += word_tokenize(sent)
    
    tokens = [t for t in tokens if t.lower() not in stopwords.words('english') and len(t) >= 3 and t.lower() != 'said']

    unique_terms = get_unique_terms(tokens)

    # get TF and IDF then calculate TF-IDF
    term_freqs = get_term_frequency(unique_terms, tokens)
    idf = get_inverse_document_frequency(content, unique_terms)
    tfidf_scores = get_tf_idf(unique_terms, term_freqs, idf)

    # take the top 10 words with highest TF-IDF score
    # swap keys and values so the list can be sorted by TF-IDF score easily
    swapped_key_and_vals = []
    for item in tfidf_scores.items():
        swapped_key_and_vals.append((item[1], item[0]))

    # take the last ten items in reversed order so it's sorted in descending order
    top_ten = sorted(swapped_key_and_vals)[-1:-11:-1]
    top_ten_terms = [item[1] for item in top_ten]

    return ';'.join(top_ten_terms) # list should be semi-colon separated values

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
        'topic': [],
        'keywords': []
    }

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

        # find the keywords for the article using TF-IDF
        results['keywords'].append(find_keywords(df.iloc[i]['content']))

    article_nlp = pd.DataFrame(results)

    return article_nlp

def process_file(path):
    articles = pd.read_csv(path)
    nlp_result = perform_nlp(articles)
    nlp_result.to_csv('nlp_result.csv', index=False)

# read in file that was copied to this server, then pass it to perform_nlp()
# articles = pd.read_csv('NewsNLP/articles_for_nlp.csv')
# nlp_result = perform_nlp(articles)
# nlp_result.to_csv('NewsNLP/nlp_result.csv', index=False)
