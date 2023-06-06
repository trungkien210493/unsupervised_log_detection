from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import math


def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        result.append(lemmatize_stemming(token))
    return result

def calculate_score(data, vectorizer, entropy):
    """Calculate the score for new coming data

    Args:
        data (dict): The dictionary with the format: {LE name: dataframe}
        vectorizer (CountVectorizer): The trained vector
        entropy (dict): The dictionary with the format: {LE name: {term: entropy}}
    """
    score = dict()
    for key, val in data.items():
        score[key] = 0
        counts = calculate_term_vector(val, vectorizer).sum().to_dict()
        for term, count in counts.items():
            try:
                score[key] += (entropy[key][term] * math.log2(1 + count)) ** 2
            except:
                pass
        score[key] = math.sqrt(score[key])
    return score

def calculate_term_vector(data, vectorizer):
    """
    Calculate the term vector base on data
    Args:
        data (dataframe): format time, message
        vectorizer (CountVectorizer): The trained vector
    """
    # Tokenize data
    data['process'] = data['message'].map(preprocess)
    matrix = vectorizer.transform(data['process'])
    # Count term per each document
    counts = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names()).copy()
    return counts
    
def preprocess_training_data(data):
    """Calculate the entropy (et) for each logging entity (LE) in training phase

    Args:
        data (dict): The dictionary with the format: {LE name: dataframe}
    """
    result = dict()
    df = pd.concat([x for x in data.values()], ignore_index=True)
    df['process'] = df['message'].map(preprocess)
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    vectorizer.fit(df['process'])
    for key, val in data.items():
        et = preprocess_log_entities_data(val, vectorizer)
        result[key] = et
    return result, vectorizer

def calculate_entropy(x):
    if x == 0:
        return 0
    else:
        return x * math.log2(x)

def preprocess_log_entities_data(le_data, vectorizer):
    """Calculate the entropy from database normative chunks

    Args:
        le_data (dict): The logging entity data, data frame with format: time, log
        vectorizer (CountVectorizer): The trained vector
    """
    counts = calculate_term_vector(le_data, vectorizer)
    counts['timestamp'] = le_data['timestamp'].values
    counts = counts.sort_values(by='timestamp')
    # print(counts[0:20])
    # Resample data to period and sum the term occurrences
    time = counts['timestamp'].dt.to_period('10S')
    agg = counts.groupby([time]).sum()
    # print(agg[['last', 'message', 'rpd']])
    agg_df = agg.div(agg.sum(axis=0), axis=1)
    # Calculate the p * log2(p)
    agg_df = agg_df.applymap(calculate_entropy)
    # Sum according to column to get sum of all p * log2(p) (entropy of a term in M normative chunks). After that, divide M and plus 1 to calculate entropy.
    entropy = 1 + agg_df.sum()/math.log2(len(agg_df))
    return entropy.to_dict()


# if __name__ == '__main__':
#     filepath = '/home/kien/SVTECH_CODE/log_template_SVTECH/data_without_template_per_host/ME_PR02.MAP063_RE0.csv'
#     df = pd.read_csv(filepath)
#     print(preprocess_training_data({'LE1': df}))        