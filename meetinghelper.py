import streamlit as st
import pytesseract
from PIL import Image
import docx2txt
import pdfplumber
import os
import tempfile
import spacy
import gensim
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("Document Analysis App")

uploaded_files = st.file_uploader("Upload your files", type=["jpeg", "png", "jpg", "docx", "pdf", "txt"], accept_multiple_files=True)
ocr_texts = []
all_texts = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            ocr_texts.append(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
                temp_txt.write(text.encode())
                st.download_button(label="Download OCR Text", data=text, file_name=os.path.basename(temp_txt.name))
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(uploaded_file)
            all_texts.append(text)
        elif uploaded_file.type == "application/pdf":
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            all_texts.append(text)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            all_texts.append(text)

    all_texts.extend(ocr_texts)

    combined_text = " ".join(all_texts)

    if combined_text:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(combined_text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
        topics = ldamodel.print_topics(num_words=4)

        st.subheader("Key Topics")
        for idx, topic in topics:
            st.write(f"Topic {idx + 1}: {topic}")

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
        st.subheader("Word Cloud of Key Topics")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.download_button(label="Download Summary", data=combined_text, file_name="summary.txt")
