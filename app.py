from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import re
import streamlit as st
import fitz  
import nltk
import numpy as np


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words


def clean_text(text):
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'[^A-Za-z0-9.,?! ]+', '', text)  
    return text


def extract_text_from_pdf(file):
    text = ""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return clean_text(text)


def text_to_vector(text, glove_embeddings):
    words = preprocess_text(text)
    word_vectors = [glove_embeddings.get(word) for word in words if word in glove_embeddings]
    if not word_vectors:
        return np.zeros((100,))
    return np.mean(word_vectors, axis=0)


def summarize_text(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])


def find_best_answer(question, sentences, glove_embeddings):
    question_vector = text_to_vector(question, glove_embeddings)
    sentence_vectors = [text_to_vector(sentence, glove_embeddings) for sentence in sentences]
    similarities = [cosine_similarity([question_vector], [sentence_vector])[0][0] for sentence_vector in sentence_vectors]
    best_sentence_index = np.argmax(similarities)
    return sentences[best_sentence_index]


def main():
    st.title("PDF Question Answering AI")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        sentences = sent_tokenize(text)
        
        st.markdown("## Extracted Text", unsafe_allow_html=True)
        st.markdown('<div style="background-color: #D3D3D3; padding: 10px; border-radius: 5px;">' + text + '</div>', unsafe_allow_html=True)
        
        glove_embeddings = load_glove_embeddings("glove.6B.100d.txt")  
        
        if st.button("Summarize PDF"):
            summary = summarize_text(text)
            st.markdown("## Summary", unsafe_allow_html=True)
            st.markdown('<div style="background-color: #ADD8E6; padding: 15px; border-radius: 5px;">' + summary + '</div>', unsafe_allow_html=True)
        
        question = st.text_input("Enter your question")
        if st.button("Get Answer"):
            if question:
                answer = find_best_answer(question, sentences, glove_embeddings)
                st.markdown("## Answer", unsafe_allow_html=True)
                st.markdown('<div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">' + answer + '</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
