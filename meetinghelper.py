import streamlit as st
from PIL import Image
import numpy as np
import docx
import PyPDF2
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import io
import traceback

# Error logging function
def log_error(e):
    st.error(f"An error occurred: {str(e)}")
    st.error(traceback.format_exc())

# Ensure NLTK stopwords are downloaded
def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

# Load the Doctr OCR model
@st.cache_resource
def load_ocr_model():
    return ocr_predictor(pretrained=True)

# Function to process Word documents using python-docx
def process_word_document(uploaded_file):
    doc = docx.Document(uploaded_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Visualize the topics using a bar plot
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 1, figsize=(15, 8))
    axes.set_title(title)
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        axes.barh(top_features, weights, height=0.7)
        axes.set_title(f'Topic {topic_idx + 1}')
        axes.set_xlabel('Word Importance')
        axes.set_ylabel('Words')
        plt.tight_layout()

    st.pyplot(fig)

try:
    st.title("Document Summarizer and Topic Modeling App")

    # Allow users to upload multiple files
    uploaded_files = st.file_uploader(
        "Upload files",
        type=['png', 'jpg', 'jpeg', 'pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )

    text_contents = []  # List to store extracted text from all files
    stop_words = get_stopwords()  # Get NLTK stopwords
    ocr_model = load_ocr_model()  # Load the cached Doctr OCR model

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            st.write(f"Processing file: {uploaded_file.name} (type: {file_type})")

            if file_type in ["image/png", "image/jpeg"]:
                # Read the uploaded file as an image
                file_bytes = uploaded_file.read()
                try:
                    image = Image.open(io.BytesIO(file_bytes))
                    # Display the uploaded image
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                    # Convert the image to NumPy array for OCR
                    image_np = np.array(image)

                    # Run OCR on the image
                    result = ocr_model([image_np])

                    # Extract and display the OCR result
                    extracted_text = "\n".join([block['value'] for block in result.export()['pages'][0]['blocks']])
                    st.write(f"Extracted Text from {uploaded_file.name}:")
                    st.write(extracted_text)
                    text_contents.append(extracted_text)

                    # Provide option to download the extracted text
                    text_file = f"{uploaded_file.name}.txt"
                    st.download_button(
                        label=f"Download Text from {uploaded_file.name}",
                        data=extracted_text,
                        file_name=text_file,
                        mime='text/plain'
                    )
                except Exception as e:
                    log_error(e)

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
                extracted_text = process_word_document(uploaded_file)
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

            # Summarization using sumy
            def summarize_text(text, sentence_count=5):
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                summarizer = LsaSummarizer()
                summary = summarizer(parser.document, sentence_count)
                return " ".join([str(sentence) for sentence in summary])

            st.subheader("Summarization")
            try:
                summary = summarize_text(full_text)
                st.write(summary)
            except ValueError:
                st.write("Text too short to summarize.")

            # Topic Modeling using sklearn
            st.subheader("Topic Modeling")

            def preprocess(text):
                # Basic text preprocessing to remove stopwords
                vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
                transformed_data = vectorizer.fit_transform([text])
                return transformed_data, vectorizer

            # Preprocess the concatenated text
            transformed_texts, vectorizer = preprocess(full_text)

            # Fit the LDA model using sklearn
            lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
            lda_model.fit(transformed_texts)

            # Get the top words for each topic
            def display_topics(model, feature_names, no_top_words):
                for topic_idx, topic in enumerate(model.components_):
                    st.write(f"Topic {topic_idx}:")
                    st.write(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

            st.write("Topics Identified:")
            no_top_words = 10
            feature_names = vectorizer.get_feature_names_out()
            display_topics(lda_model, feature_names, no_top_words)

            # Plot the top words in each topic
            st.subheader("Topic Visualization")
            plot_top_words(lda_model, feature_names, no_top_words, "Topics in LDA Model")

except Exception as e:
    log_error(e)
