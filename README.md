# Simple RAG System with PDFs/Text, Weaviate, LangChain & Tavily Tool

## Overview

This is a simple Retrieval-Augmented Generation (RAG) chatbot app that:

- Takes PDF or text documents as input
- Embeds their content into a **Weaviate** vector store
- Retrieves relevant context from Weaviate at query time
- Uses a **LangChain agent** for reasoning and generation
- Integrates a **Tavily tool** to enable modern, natural language questions

---

## Components

| Component            | Purpose                                          |
|----------------------|-------------------------------------------------|
| PDF/Text ingestion   | Parse and extract text from documents           |
| Embedding generation | Convert text into vector embeddings              |
| Vector store         | Store embeddings and enable similarity search   |
| LangChain agent      | Orchestrate retrieval + language model generation |
| Tavily tool          | External tool/plugin to enhance chatbot queries |

---

## How to Run

### 1. Insert data
Create a folder name "data" or alike and insert either .txt or pdf files for data. Modify the script to support other types like .csv if necessary

### 2. Run Desktop and Web versions  
Run the `main.py` script to launch a local desktop application with a Tkinter GUI.  
Run this 'streamlit run streamlit_app.py' in terminal to launch a web application via Streamlit

```bash
python main.py
streamlit run streamlit_app.py

