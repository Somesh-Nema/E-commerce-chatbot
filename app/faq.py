import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Load CSV path
faqs_path = Path(__file__).parent / "resource/faq_data.csv"

# ✅ Initialize ChromaDB and Groq
chroma_client = chromadb.Client()
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

collection_name_faqs = 'faqs'

# ✅ Load embedding model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ✅ Ingest FAQ CSV
def ingest_faq_data(path):
    if collection_name_faqs not in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.create_collection(
            name=collection_name_faqs,
            embedding_function=ef
        )
        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]

        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
    else:
        print(f"Collection '{collection_name_faqs}' already exists.")

# ✅ Query top relevant answer
def get_relevant_qa(query):
    collection = chroma_client.get_collection(
        name=collection_name_faqs,
        embedding_function=ef
    )
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

# ✅ Get answer with context
def faq_chain(query):
    result = get_relevant_qa(query)
    # Extract the 'answer' field from each metadata
    context = '\n'.join([meta.get('answer', '') for meta in result['metadatas'][0]])
    return generate_answer(query, context)

# ✅ Generate answer using GROQ
def generate_answer(query, context):
    prompt = f'''
Given the question and context below, generate the answer based on the context only.
If you don't find the answer in it then say "I don't know". Do not make things up.

QUESTION: {query}
CONTEXT: {context}
    '''

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=os.environ['GROQ_MODEL'],  # Make sure GROQ_MODEL is set in .env
    )

    return chat_completion.choices[0].message.content

# ✅ Main flow
if __name__ == "__main__":
    ingest_faq_data(faqs_path)
    query = "do you take cash as a payment option?"
    answer = faq_chain(query)
    print(answer)
