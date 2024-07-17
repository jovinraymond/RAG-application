
from flask import Flask, request, render_template
import sqlite3
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

app = Flask(__name__)

# Initialize OpenAI API key
openai.api_key = 'API KEY'

# Load vectorizer and database
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

db_name = 'mental_health_corpus.db'
table_name = 'documents'

def vectorize_query(query, vectorizer):
    query_vector = vectorizer.transform([query])
    return query_vector

def retrieve_documents(query, vectorizer, db_name, table_name, top_n=3):
    query_vector = vectorize_query(query, vectorizer)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, text, vector FROM {table_name}")
    rows = cursor.fetchall()

    doc_scores = []
    for row in rows:
        doc_id, text, vector_blob = row
        doc_vector = pickle.loads(vector_blob)
        similarity = cosine_similarity(query_vector, doc_vector)
        doc_scores.append((similarity, text))

    doc_scores.sort(reverse=True, key=lambda x: x[0])
    top_documents = [doc[1] for doc in doc_scores[:top_n]]
    conn.close()
    return top_documents

def generate_response_with_gpt3(query, documents):
    prompt = f"Q: {query}\n"
    for i, doc in enumerate(documents, 1):
        prompt += f"Context {i}: {doc}\n"
    prompt += "A:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7
    )

    answer = response['choices'][0]['message']['content'].strip()
    return answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        top_documents = retrieve_documents(query, vectorizer, db_name, table_name)
        response = generate_response_with_gpt3(query, top_documents)
        return render_template('index.html', query=query, response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
