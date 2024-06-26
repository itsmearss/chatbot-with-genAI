# Vertex
import os

from flask import Flask, render_template, request

import vertexai
import vertexai.preview.generative_models as generative_models

from vertexai.preview.generative_models import GenerativeModel

# Gen AI
import pathlib
import textwrap

import PIL.Image

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

#pylint: disable=C0103
app = Flask (__name__)

# Gen AI
# Replace with your api key
GOOGLE_API_KEY='YOUR API KEY'
genai.configure(api_key=GOOGLE_API_KEY)

model_genai_text = genai.GenerativeModel('gemini-1.0-pro')
model_genai_image = genai.GenerativeModel('gemini-pro-vision')
model_genai_chat = genai.GenerativeModel('gemini-pro')

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('prompt')
        model_responses = generate(user_input)
        
        response = ""
        for r in model_responses:
            response += r.text
            
        return {
            'answer': response
        }
    else:
        return {
            'answer': 'not found'
        }
       
@app.route('/genai-text', methods=['GET', 'POST'])
def genai():
    if request.method == 'POST':
        user_input = request.form.get('prompt')
        model_responses = model_genai_text.generate_content(user_input)
            
        return {'answer': model_responses.text}
    else:
        return {
            'answer': 'not found'
        }
       
@app.route('/genai-image', methods=['GET', 'POST'])
def genaiImage():
    if request.method == 'POST':
        user_input = request.form.get('prompt')
        image = request.files['image']
        img = PIL.Image.open(image)
        
        model_responses = model_genai_image.generate_content([user_input, img], stream=False)
            
        return {'answer': model_responses.text}
    else:
        return {
            'answer': 'not found'
        }

@app.route('/genai-chat', methods=['GET', 'POST'])
def genaiChat():
    if request.method == 'POST':
        user_input = request.form.get('prompt')
        
        chat = model_genai_chat.start_chat(history=[])
        
        model_responses = chat.send_message(user_input)
            
        return {'answer': model_responses.text}
    else:
        return {
            'answer': 'not found'
        }

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')