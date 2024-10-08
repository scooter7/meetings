import streamlit as st
from PIL import Image
import pytesseract
import docx2txt
import PyPDF2
import easyocr
import numpy as np
import io
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import streamlit.components.v1 as components
import nltk
from nltk.corpus import stopwords
from gensim.summarization import summarize

# Initialize NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.title("Document Summarizer and Topic Modeling App")

# Allow users to upload multiple files
uploaded_files = st.file_uploader(
    "Upload files",
    type=['png', 'jpg', 'jpeg', 'pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

text_contents = []  # List to store extracted text from all files

if uploaded_files:
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        st.write(f"Processing file: {uploaded_file.name} (type: {file_type})")

        if file_type in ["image/png", "image/jpeg"]:
            # Process images
            image = Image.open(uploaded_file)
            result = reader.readtext(np.array(image))
            extracted_text = ' '.join([text[1] for text in result])
            st.write(f"Extracted Text from {uploaded_file.name}:")
            st.write(extracted_text)
            text_contents.append(extracted_text)

            # Provide option to download the text
            text_file = f"{uploaded_file.name}.txt"
            st.download_button(
                label=f"Download Text from {uploaded_file.name}",
                data=extracted_text,
                file_name=text_file,
                mime='text/plain'
            )

        elif file_type == "application/pdf":
            # Process PDFs
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            extracted_text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()
            st.write(f"Extracted Text from {uploaded_file.name}:")
            st.write(extracted_text)
            text_contents.append(extracted_text)

        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Process Word documents
            extracted_text = docx2txt.process(uploaded_file)
            st.write(f"Extracted Text from {uploaded_file.name}:")
            st.write(extracted_text)
            text_contents.append(extracted_text)

        elif file_type == "text/plain":
            # Process text files
            extracted_text = uploaded_file.read().decode('utf-8')
            st.write(f"Content of {uploaded_file.name}:")
            st.write(extracted_text)
            text_contents.append(extracted_text)

        else:
            st.write(f"Unsupported file type: {file_type}")

    if text_contents:
        # Concatenate all text contents
        full_text = ' '.join(text_contents)

        # Summarization using Gensim
        st.subheader("Summarization")
        try:
            summary = summarize(full_text, ratio=0.2)
            st.write(summary)
        except ValueError:
            st.write("Text too short to summarize.")

        # Topic Modeling
        st.subheader("Topic Modeling")
        # Preprocess text
        def preprocess(text):
            tokens = gensim.utils.simple_preprocess(text, deacc=True)
            return [token for token in tokens if token not in stop_words]

        processed_texts = [preprocess(text) for text in text_contents]
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        lda_model = gensim.models.LdaModel(
            corpus, num_topics=5, id2word=dictionary, passes=15
        )

        st.subheader("Topics Identified:")
        topics = lda_model.print_topics()
        for topic in topics:
            st.write(topic)

        # Visualize the topics
        st.subheader("Topic Modeling Visualization")
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, 'lda.html')

        # Display the visualization in Streamlit
        HtmlFile = open("lda.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=800, scrolling=True)
