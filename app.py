import os
import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient

# Hugging Face Model Details
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# Validate API Key
if not HF_API_KEY:
    st.error("Hugging Face API Key is missing. Add it to GitHub Secrets or set it locally.")
    st.stop()

# Initialize Hugging Face Inference Client
client = InferenceClient(model=HF_MODEL, token=HF_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip() if text else "No readable text found in the PDF."

# Function to extract text from a website
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text() for p in paragraphs])
        return text[:3000] if text else "No text found on the page."
    except requests.exceptions.RequestException:
        return "Failed to fetch the webpage. Check the URL."

# Function to generate summary
def generate_summary(text):
    response = client.text_generation(f"Summarize this: {text}", max_new_tokens=200)
    
    # Check if response is a string (error handling)
    if isinstance(response, str):
        return response  # Likely an error message from the API
    
    return response  # Response is already a string

# Function to generate quiz questions
def generate_quiz(text):
    response = client.text_generation(f"Generate 5 multiple-choice questions from this text: {text}", max_new_tokens=300)
    
    # Check if response is a string (error handling)
    if isinstance(response, str):
        return response  # Likely an error message from the API
    
    return response  # Response is already a string

# Streamlit UI
st.title("📚 AI Tutor: Personalized Learning Assistant")

option = st.radio("Choose Input Type:", ("Upload PDF", "Enter Text", "Website URL"))

input_text = ""

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        input_text = extract_text_from_pdf(uploaded_file)

elif option == "Enter Text":
    input_text = st.text_area("Paste your text here:")

elif option == "Website URL":
    url = st.text_input("Enter the website URL:")
    if url:
        input_text = extract_text_from_url(url)

if st.button("Generate Summary & Quiz") and input_text:
    with st.spinner("Generating content..."):
        st.subheader("📌 Summary")
        st.write(generate_summary(input_text))
        
        st.subheader("📝 Quiz Questions")
        st.write(generate_quiz(input_text))

st.markdown("\n---\n👨‍💻 Built with LLaMA & Streamlit")
