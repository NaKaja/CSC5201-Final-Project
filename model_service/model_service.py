from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram
from prometheus_client.exposition import generate_latest
import torch
import nltk
import time


class TransformerPredict:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BartForConditionalGeneration.from_pretrained('kajan1/bart-large-cnn-khan')
        self.model.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained('kajan1/bart-large-cnn-khan')

    """Uses chunking to split document by sentences and regroup into chunks that are < max model input
       Summarizes each chunk, concatenates separated by newlines, and returns the summaries as one string

        Args:
            text (str): The text to be summarized
            min_len (int): The minimum length of the output
            max_len (int): The maximum length of the output
            beams (int): Number of beams to use with beam search decoding
            sample (bool): Whether the model uses sampling in decoding
    """

    def summarize(self, text, min_len=0, max_len=128, beams=1, sample=False):
        # Split document into sentences
        sentences = nltk.tokenize.sent_tokenize(text)
        length = 0
        chunk = ""
        chunks = []

        # Aggregate sentences into chunks that are < model_max_length
        for i, sen in enumerate(sentences):
            combined_length = len(self.tokenizer.tokenize(sen)) + length
            if combined_length <= self.tokenizer.max_len_single_sentence:
                chunk += sen + " "
                length = combined_length
                if i == len(sentences) - 1:
                    chunks.append(chunk.strip())
            else:
                chunks.append(chunk.strip())
                chunk = sen + " "
                length = len(self.tokenizer.tokenize(sen))

        # Generate and decode summaries for each chunk
        res = ""
        for i in [self.tokenizer(c, return_tensors='pt') for c in chunks]:
            summary_ids = self.model.generate(i["input_ids"].to(self.device),
                                              num_beams=beams,
                                              min_length=min_len,
                                              max_length=max_len,
                                              do_sample=sample)
            summary = self.tokenizer.batch_decode(summary_ids,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0]
            res += summary + "\n"
        return res[:-1]


app = Flask(__name__)
summarizer = TransformerPredict()
REQUEST_COUNT = Counter('request_count', 'App Request Count', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_ms', 'Request latency in milliseconds', ['method', 'endpoint'])


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    params = data['params']
    summary = summarizer.summarize(input_text, **params)
    return jsonify({'summary': summary})


@app.before_request
def before_request():
    request.start_time = round(time.time() * 1000)


@app.after_request
def after_request(response):
    request_latency = round(time.time() * 1000) - request.start_time
    print(request_latency)
    REQUEST_COUNT.labels(request.method, request.path).inc()
    REQUEST_LATENCY.labels(request.method, request.path).observe(request_latency)
    return response


@app.route('/metrics')
def metrics():
    return generate_latest()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
