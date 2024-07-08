import streamlit as st
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline

# Load the pre-trained BERT model for question-answering
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Define a username and password for demonstration purposes
valid_username = "user"
valid_password = "password"

# Streamlit web app
st.title("HR Hive Bot")

# Create a sidebar for navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "FAQ", "Contact", "Login"])

if page == "Home":
    # Input field for the context
    file_path = "dataset.txt"
    try:
        with open(file_path, "r") as file:
            text = file.read()
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    context = text

    # Input field for the question
    question = st.text_input("Ask a question:", "What is an internship?")

    # Answer button
    if st.button("Get Answer"):
        if context and question:
            # Get the answer using the question-answering model
            answer = nlp({
                'question': question,
                'context': context
            })

            # Display the answer
            st.write("Answer:", answer['answer'])

elif page == "FAQ":
    st.header("Frequently Asked Questions")
    # Add FAQ content here
    # You can use st.write, st.markdown, st.table, etc. to display your FAQ content.

elif page == "Contact":
    st.header("Contact Us")
    # Add contact information or a contact form here

elif page == "Login":
    st.header("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == valid_username and password == valid_password:
            st.success("Login Successful!")
            # You can add code here to grant access to protected pages
        else:
            st.error("Login Failed. Please check your credentials.")

# Optionally, you can add more pages and customize the layout as needed.
