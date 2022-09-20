from flask import Flask, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone




app = Flask(__name__)

def query_pinecone(query, model, index, top_k=50):
    query_emb = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    res = index.query(query_emb.tolist(), top_k=top_k, include_metadata=True)

    return res.to_dict()

@app.route('/', methods=['GET'])
def index():
    return "Welcome to NOCD Search!"

@app.route('/search', methods=['POST'])
def search():
    if request.method=="POST":
        payload = request.get_json()
        query = payload.get('query')

        prediction = query_pinecone(query=query, model=model, index=index)
        
        return jsonify({'results': prediction['matches']})
    return "Not a proper request method or data"


if __name__ == '__main__':

    model = SentenceTransformer('msmarco-distilbert-base-tas-b', device='cpu')
    model.max_seq_length = 256

    with open('secrets', 'r') as fp:
        API_KEY = fp.read()  # get api key app.pinecone.io

    pinecone.init(
        api_key=API_KEY,
        environment='us-west1-gcp'
    )

    index = pinecone.Index('nocd-search')


    app.run(debug=True, host='0.0.0.0', port=5000)