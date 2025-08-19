# Simple RAG System with PDFs/Text, Weaviate, LangChain & Tavily Tool

## Overview

This is a simple Retrieval-Augmented Generation (RAG) chatbot app that:

- Takes PDF or text documents as input
- Extracts **chunks of text** from PDFs for downstream tasks
- Provides MCP tools to **summarize PDFs** and **search for keywords/snippets** locally
- Embeds their content into a **Weaviate** vector store
- Retrieves relevant context from Weaviate at query time
- Uses a **LangChain agent** for reasoning and generation
- Integrates a **Tavily tool** to enable modern, natural language web queries

---

## Components

| Component              | Purpose                                                     |
|------------------------|-------------------------------------------------------------|
| PDF/Text ingestion     | Parse and extract text from documents                       |
| PDF summarization      | Generate concise summaries of uploaded PDF content          |
| PDF keyword search     | Find relevant pages/snippets within PDFs using keywords     |
| Embedding generation   | Convert text into vector embeddings                         |
| Vector store           | Store embeddings and enable similarity search               |
| LangChain agent        | Orchestrate retrieval + language model generation           |
| Tavily tool            | External tool/plugin to enhance chatbot queries with the web|

---

## How to Run

### 1. Insert data
Create a folder named **`data`** (or similar) and insert either `.txt` or `.pdf` files for ingestion.  
Modify the script to support other types like `.csv` if necessary.

### 2. Run Desktop and Web versions  
Run the `main.py` script to launch a local desktop application with a Tkinter GUI.  
Or run the Streamlit version in your terminal:

```bash
python main.py
streamlit run streamlit_app.py

