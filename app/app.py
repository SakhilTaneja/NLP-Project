# app.py
import streamlit as st
from classify_model import classify
from summary_model import summarize
import fitz  # PyMuPDF

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Set up the Streamlit interface
st.title("Text Analysis with LangChain")
st.subheader("Classify and Summarize Text from PDF")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted from PDF:")
        st.write(extracted_text)

        # Classification
        st.header("Classification")
        if st.button("Classify"):
            if extracted_text:
                with st.spinner("Classifying..."):
                    classification = classify(extracted_text)
                    st.success("Classification Result:")
                    st.write(f"Class: {classification}")
                    
                    # If classification is 1, proceed to summarization immediately
                    if classification == 1:
                        st.write("This is a Successful Case, Begining Summarization")
                        st.header("Summarization")
                        with st.spinner("Summarizing..."):
                            summary = summarize(extracted_text)
                            st.success("Summary:")
                            st.write(summary)
                    else:
                        st.warning("This is not a Successful Case, skipping summarization.")
            else:
                st.warning("Please upload a PDF to classify.")
else:
    st.warning("Please upload a PDF file.")