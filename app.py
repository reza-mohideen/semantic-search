from flask import Flask, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
import os




app = Flask(__name__)

def query_pinecone(embed, index, top_k=50):
    
    res = index.query(embed, top_k=top_k, include_metadata=True)

    return res.to_dict()

def huggingface_embed(query, model):
    query_emb = model.encode(query, convert_to_tensor=True, show_progress_bar=False)

    return query_emb.tolist()

def openai_embed(query):
    import openai

    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai_model_query = 'text-search-babbage-query-001'
    res = openai.Embedding.create(input=query, engine=openai_model_query)
    embed = [record['embedding'] for record in res['data']]

    return embed[0]

@app.route('/', methods=['GET'])
def index():
    return "Welcome to NOCD Search!"

@app.route('/search-huggingface', methods=['POST'])
def search_huggingface():
    if request.method=="POST":
        payload = request.get_json()
        query = payload.get('query')
        
        embed = huggingface_embed(query=query, model=model)
        res = query_pinecone(embed=embed, index=huggingface_index)
        
        return jsonify({'results': res['matches']})
    return "Not a proper request method or data"

@app.route('/search-openai', methods=['POST'])
def search_openai():
    if request.method=="POST":
        payload = request.get_json()
        query = payload.get('query')
        
        embed = openai_embed(query=query)
        res = query_pinecone(embed=embed, index=openai_index)
        
        return jsonify({'results': res['matches']})
    return "Not a proper request method or data"


if __name__ == '__main__':

    print('Loading model...')
    model = SentenceTransformer('msmarco-distilbert-base-tas-b', device='cpu')
    model.max_seq_length = 256

    print('Connecting to Pinecone...')
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment='us-west1-gcp'
    )

    huggingface_index = pinecone.Index('nocd-search-huggingface')
    openai_index = pinecone.Index('nocd-search-openai')


    app.run(debug=True, host='0.0.0.0', port=5000)
    