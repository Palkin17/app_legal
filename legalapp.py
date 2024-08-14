# Import necessary modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st
import fitz  # PyMuPDF for PDF processing

# Define a prompt template for the chatbot
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions based on the provided document."),
        ("user", "Question:{question}\nDocument Content:{content}")
    ]
)

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please provide a summary of the provided document."),
        ("user", "Document Content:{content}")
    ]
)

key_terms_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please extract key terms from the provided document."),
        ("user", "Document Content:{content}")
    ]
)

# Set up the Streamlit framework with multiple pages
st.title('Legal AI - POC ')  # Set the title of the Streamlit app

# Add an image to the first page
st.image("M&A.png", use_column_width=True)

# Create a sidebar for the file uploader
st.sidebar.markdown("<h4 style='color:#4B8BBE;'>Upload PDF</h4>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Choose your PDF file", type="pdf")

# Add a separator in the sidebar
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Create a sidebar for navigation
st.sidebar.markdown("<h2 style='color:#4B8BBE;'>Navigation</h2>", unsafe_allow_html=True)
chat_page = st.sidebar.button("Chat with Your Data")
summary_page = st.sidebar.button("Summary of Your Data")
key_terms_page = st.sidebar.button("Key Term Extraction")

# Initialize the Ollama model
llm = Ollama(model="llama2")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

# Process the uploaded file
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("PDF Content Extracted")

    if chat_page:
        st.header("Chat with Your Data")
        # Create a chain that combines the prompt and the Ollama model
        chain = chat_prompt | llm

        # User input for questions
        input_text = st.text_input("Ask your question!")

        # Invoke the chain with the input text and the extracted PDF content
        if input_text:
            response = chain.invoke({"question": input_text, "content": pdf_text})
            st.write(response)

    if summary_page:
        st.header("Summary of Your Data")
        # Create a chain for summarizing the document
        chain = summary_prompt | llm

        # Get the summary of the document
        if pdf_text:
            summary = chain.invoke({"content": pdf_text})
            st.write(summary)

    if key_terms_page:
        st.header("Key Term Extraction")
        # Create a chain for extracting key terms from the document
        chain = key_terms_prompt | llm

        # Extract key terms from the document
        if pdf_text:
            key_terms = chain.invoke({"content": pdf_text})
            st.write("Key Terms:", key_terms)
