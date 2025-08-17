import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words_id = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer_id = factory.create_stemmer()

def lowering(text: str) -> str:
    return text.lower()

def remove_punctuation_and_symbol(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def stopword_removal(text: str) -> str:
    return " ".join([word for word in text.split() if word not in stop_words_id])

def stemming(text: str) -> str:
    return stemmer_id.stem(text)

# def remove_html_tags(text: str) -> str:
#     return re.sub(r'<[^>]+>', '', text)

def preprocessing_id(text: str, do_stemming: bool=True) -> str:
    if not isinstance(text, str):
        return ""
    # if remove_html:
    #     text = remove_html_tags(text)
    text = lowering(text)
    text = remove_punctuation_and_symbol(text)
    text = stopword_removal(text)
    if do_stemming:
        text = stemming(text)
    return text