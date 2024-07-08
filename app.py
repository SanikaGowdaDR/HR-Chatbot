from flask import Flask, render_template, request
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import pipeline
from utils import split_into_chunks

app = Flask(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# model loading and setup
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, device=device)


# Dummy context for demonstration
file_path = "dataset_win.txt"
f = open(file_path, "r")
text = f.read()


text = text.replace("\n", " ")

chunks = split_into_chunks(text, tokenizer)

# Define routes
# @app.route('/')
# def home():
#     return render_template('home.html', context=context)

# @app.route('/get_answer', methods=['POST'])
# def get_answer():
#     question = request.form['question']

#     if context and question:
#         # Get the answer using the question-answering model
#         answer = nlp({
#             'question': question,
#             'context': context
#         })
#         return render_template('home.html', answer=answer['answer'])
#     else:
#         return "Context or question is missing. Please provide both."

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None

    if request.method == 'POST':
        question = request.form['question']

        if chunks and question:
            questions = [question for _ in range(len(chunks))]
            # Get the answer using the question-answering model
            inputs = {"question": questions, "context": chunks}
            outputs = qa_pipeline(inputs)
            best_answer = None
            for ans in outputs:
                if best_answer is None or best_answer["score"] <= ans["score"]:
                    best_answer = ans
            print(best_answer)
            answer = best_answer

    return render_template('home.html', answer=answer)

@app.route('/faq')
def faq():
    # Add FAQ content here
    return render_template('faq.html')

@app.route('/about')
def about():
    # Add FAQ content here
    return render_template('about.html')

@app.route('/contact')
def contact():
    # Add contact information or a contact form here
    return render_template('contact.html')


if __name__ == '__main__':
    app.run()
