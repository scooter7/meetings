import streamlit as st
from PIL import Image
import numpy as np
import docx
import PyPDF2
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import io
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the Doctr OCR model
@st.cache_resource
def load_ocr_model():
    return ocr_predictor(pretrained=True)

# Process Word documents using python-docx
def process_word_document(uploaded_file):
    doc = docx.Document(uploaded_file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Summarization using spaCy
def summarize_text(text, sentence_count=5):
    doc = nlp(text)
    # Extract top sentences for summarization (by length here, can be tweaked)
    sentences = list(doc.sents)
    return ' '.join([str(sent) for sent in sentences[:sentence_count]])

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

# Main function
def main():
    st.title("Document Summarizer and Topic Modeling App")

    # Allow users to upload multiple files
    uploaded_files = st.file_uploader(
        "Upload files", 
        type=['png', 'jpg', 'jpeg', 'pdf', 'docx', 'txt'], 
        accept_multiple_files=True
    )

    text_contents = []  # Store extracted text from all files
    ocr_model = load_ocr_model()  # Load the cached Doctr OCR model

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            st.write(f"Processing file: {uploaded_file.name} (type: {file_type})")

            if file_type in ["image/png", "image/jpeg"]:
                try:
                    image = Image.open(io.BytesIO(uploaded_file.read()))
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    image_np = np.array(image)
                    result = ocr_model([image_np])
                    ocr_output = result.export()

                    extracted_text = []
                    for page in ocr_output.get('pages', []):
                        for block in page.get('blocks', []):
                            extracted_text.append(block.get('value', block.get('text', '')))
                    
                    extracted_text = "\n".join(extracted_text)
                    st.write(f"Extracted Text from {uploaded_file.name}:")
                    st.write(extracted_text)
                    text_contents.append(extracted_text)

                    st.download_button(
                        label=f"Download Text from {uploaded_file.name}",
                        data=extracted_text,
                        file_name=f"{uploaded_file.name}.txt",
                        mime='text/plain'
                    )

                except Exception as e:
                    st.error(f"An error occurred while processing the image: {e}")

            elif file_type == "application/pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    extracted_text = ''.join(page.extract_text() for page in pdf_reader.pages)
                    st.write(f"Extracted Text from {uploaded_file.name}:")
                    st.write(extracted_text)
                    text_contents.append(extracted_text)
                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {e}")

            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                extracted_text = process_word_document(uploaded_file)
                st.write(f"Extracted Text from {uploaded_file.name}:")
                st.write(extracted_text)
                text_contents.append(extracted_text)

            elif file_type == "text/plain":
                extracted_text = uploaded_file.read().decode('utf-8')
                st.write(f"Content of {uploaded_file.name}:")
                st.write(extracted_text)
                text_contents.append(extracted_text)

            else:
                st.write(f"Unsupported file type: {file_type}")

        if text_contents:
            full_text = ' '.join(text_contents)

            # Summarization
            st.subheader("Summarization")
            try:
                summary = summarize_text(full_text)
                st.write(summary)
            except ValueError:
                st.write("Text too short to summarize.")

            # Topic Modeling
            st.subheader("Topic Modeling")

            def preprocess(text):
                vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
                transformed_data = vectorizer.fit_transform([text])
                return transformed_data, vectorizer

            transformed_texts, vectorizer = preprocess(full_text)
            lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
            lda_model.fit(transformed_texts)

            def display_topics(model, feature_names, no_top_words):
                for topic_idx, topic in enumerate(model.components_):
                    st.write(f"Topic {topic_idx}:")
                    st.write(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

            no_top_words = 10
            feature_names = vectorizer.get_feature_names_out()
            display_topics(lda_model, feature_names, no_top_words)

            plot_top_words(lda_model, feature_names, no_top_words, "Topics in LDA Model")

if __name__ == "__main__":
    main()
