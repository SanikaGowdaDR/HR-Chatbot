import streamlit as st
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline


# Load the pre-trained BERT model for question-answering
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)


# Streamlit web app
st.title("HR Hive bot")

# Input field for the context
#context = st.text_area("Enter the context:", "An internship is...")

file_path = "dataset.txt"

try:
    with open(file_path, "r") as file:
        # Read the entire content of the file
        text = file.read()
        #print("Text from the file:")
        #print(text)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")


context = text
# context = ("An internship is a valuable and immersive learning experience that provides students or individuals with the opportunity to gain real-world work experience in their chosen field. During an internship, participants have the chance to apply their academic knowledge to practical situations, develop essential skills, and build a professional network. Internships can be found in various industries, ranging from business and technology to healthcare and the arts. These experiences offer a unique insight into the daily operations of a company or organization and often serve as a stepping stone for future career opportunities. Internships are not only a great way to enhance one's resume but also a crucial means of personal and professional growth, allowing individuals to explore their interests and determine if a specific career path aligns with their aspirations.")

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

# Optionally, you can add more Streamlit elements, customize the layout, a


