from PyPDF2 import PdfReader  # Import PdfReader directly

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as pdf_file_obj:
        pdf_reader = PdfReader(pdf_file_obj)  # Use PdfReader directly, no need to reference PyPDF2
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()  # Use extract_text() method
    return text

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    return tokens

def split_text_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_sentence_importance(sentences):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    importance = tfidf.toarray().sum(axis=1)
    return importance

def form_summary(sentences, importance, num_sentences):
    ranked_sentences = sorted(zip(sentences, importance), key=lambda x: x[1], reverse=True)
    summary = ' '.join([sentence for sentence, _ in ranked_sentences[:num_sentences]])
    return summary

def pdf_summarizer(file_path, num_sentences):
    text = extract_text_from_pdf(file_path)
    tokens = preprocess_text(text)
    sentences = split_text_into_sentences(text)
    importance = calculate_sentence_importance(sentences)
    summary = form_summary(sentences, importance, num_sentences)
    return summary

file_path = "C:\\Users\\hp\\Downloads\\NACo-SocialMedia-Guides.pdf"
num_sentences = 5
summary = pdf_summarizer(file_path, num_sentences)
print(summary)
input()

