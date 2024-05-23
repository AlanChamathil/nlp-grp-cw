from datetime import datetime
import logging
import sys
import io
from flask import Flask, jsonify, render_template, request, redirect, url_for
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

# pipe = pipeline("token-classification", model = "./trainedmodel/test", tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased"))
pipe = pipeline("token-classification", model="george6/roberta-finetuned-NER")

# Ensure stdout and stderr use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set up logging with UTF-8 encoding
logging.basicConfig(
    filename='logs/app.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    #encoding='utf-8'  # Ensure the log file uses UTF-8 encoding
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

# Explicitly set encoding for handlers
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setStream(io.TextIOWrapper(open(handler.baseFilename, handler.mode), encoding='utf-8'))
    elif isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
        handler.setStream(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'))

def requestResults(text):
    text = text.decode("utf-8")
    # app.logger.info(text)
    prediction = pipe(text)

    # filter out brackets and join subwords
    exclude_tokens_filtered_list = []
    filtered_list = []
    exclude_tokens = ['(', ')', '[', ']']

    for i, tags in enumerate(prediction):
        if (any(token in tags['word'] for token in exclude_tokens)) and (tags['word'].startswith('Ġ')):
            if tags['word'].startswith('Ġ'):
                # last
                if len(prediction) - 1 == i:
                    continue
                elif prediction[i + 1]['word'].startswith('Ġ'):
                    continue
                else:
                    prediction[i + 1]['word'] = "Ġ" + prediction[i + 1]['word']
        else:
            exclude_tokens_filtered_list.append(tags)

    for tags in exclude_tokens_filtered_list:
        if tags['word'].startswith('Ġ'):
            filtered_list.append(tags)

    return filtered_list

def log_interaction(input_text, filtered_list):
    with open("logs/interaction_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now()}; {input_text}; {filtered_list}\n")

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        text = request.get_data()
        # app.logger.info("Hello")
        data = requestResults(text)

        ner_tags_list = []

        # app.logger.info(data[0])
        for tag in data:
            # app.logger.info(tag['entity'])
            ner_tags_list.append(tag['entity'])
        log_interaction(text.decode("utf-8"), ner_tags_list)
        return jsonify(ner_tags_list)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
