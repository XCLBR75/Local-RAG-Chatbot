import ollama
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from typing import List
import os
import uuid

class OllamaEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [ollama.embed(model=self.model_name, input=t)['embeddings'][0] for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return ollama.embed(model=self.model_name, input=query)['embeddings'][0]



# Load the dataset

dataset = []
with open('cat-facts.txt', encoding="utf-8") as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')


# Implement the retrieval system

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Initialize embedding and Chroma vectorstore
embedding = OllamaEmbedding(EMBEDDING_MODEL)
persist_path = "chroma_cat_facts"

if os.path.exists(persist_path):
    print("Loading existing Chroma DB...")
    vectorstore = Chroma(
        collection_name="cat_facts",
        embedding_function=embedding,
        persist_directory=persist_path,
    )
else:
    print("Creating new Chroma DB...")
    vectorstore = Chroma(
        collection_name="cat_facts",
        embedding_function=embedding,
        persist_directory=persist_path,
    )

    # Populate Chroma with documents
    print("Adding data to Chroma...")
    texts = [chunk.strip() for chunk in dataset]
    ids = [str(uuid.uuid4()) for _ in texts]
    vectorstore.add_texts(texts=texts, ids=ids)
    vectorstore.persist()
    print(f"Inserted {len(texts)} entries into Chroma")


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve(query: str):
    docs = retriever.get_relevant_documents(query)
    return [(doc.page_content, 1.0) for doc in docs]  # we drop actual similarity for now


# Chatbot

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
# print(instruction_prompt)

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

