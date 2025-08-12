import os
import logging
import uuid
from typing import Tuple, Dict
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.config import Configure
from langchain_weaviate.vectorstores import WeaviateVectorStore
from embeddings import OllamaEmbedding
from PyPDF2 import PdfReader
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
executor = ThreadPoolExecutor()

def deterministic_id(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

def load_text(file_path: str) -> list[str]:
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return [line.strip() for line in text.splitlines() if line.strip()]
    else:
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

async def async_exists(collection, uid: str) -> bool:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(executor, collection.data.exists, uid)
    except Exception as e:
        logging.warning(f"Exist check failed for {uid}: {e}")
        return False

async def inject_dataset(file_path: str, topic: str) -> Tuple[Dict[str, WeaviateVectorStore], weaviate.WeaviateClient]:
    lines = load_text(file_path)
    logging.info(f"Loaded {len(lines)} lines from {file_path}")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    assert client.is_ready(), "Weaviate not ready"

    try:
        collection = client.collections.get(topic)
        logging.info(f"Using existing collection '{topic}'")
    except WeaviateBaseError:
        logging.info(f"Collection '{topic}' does not exist — creating it.")
        collection = client.collections.create(
            name=topic,
            properties=[{"name": "content", "dataType": "text"}],
            vectorizer_config=Configure.Vectorizer.none(),
        )

    embedding = OllamaEmbedding("nomic-embed-text")
    vectorstore = WeaviateVectorStore(client=client, index_name=topic, text_key="content", embedding=embedding)

    tasks = []
    uids = []
    for line in lines:
        uid = deterministic_id(line)
        tasks.append(async_exists(collection, uid))
        uids.append((line, uid))

    results = await asyncio.gather(*tasks)

    dedup_lines = []
    dedup_ids = []

    for (line, uid), exists in zip(uids, results):
        if not exists:
            dedup_lines.append(line)
            dedup_ids.append(uid)
        else:
            logging.debug(f"Skipped duplicate: {uid}")

    if dedup_lines:
        vectorstore.add_texts(texts=dedup_lines, ids=dedup_ids)
        logging.info(f"Injected {len(dedup_lines)} new records.")
    else:
        logging.info("No new lines to inject — all items existed already.")

    return {topic: vectorstore}, client
