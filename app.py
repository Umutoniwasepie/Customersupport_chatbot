import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

# Load the model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    # Replace with your actual Hugging Face model repository path
    model_name = "Umutoniwasepie/final_model"  # Update this!
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Text normalization function
def normalize_input(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?.,!]', '', text)
    return text

# Response capitalization function
def capitalize_response(response):
    sentences = response.split(". ")
    unique_sentences = []
    for s in sentences:
        if s and s not in unique_sentences:
            unique_sentences.append(s.capitalize())
    return ". ".join(unique_sentences)

# Response generation function
def generate_response(query):
    query_lower = normalize_input(query)
    input_text = f"generate response: Current query: {query_lower}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            top_k=70,
            repetition_penalty=1.5,
            do_sample=True
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return capitalize_response(response)

# Streamlit app layout
st.title("Customer Support Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input at the bottom
user_input = st.chat_input("Type your question here...")

# Handle user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = generate_response(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
