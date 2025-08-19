from injection import inject_dataset
from retrieval import setup_tools_async, create_agent
from langchain.llms import Ollama as LangchainOllama
from retrieval import setup_tools_async, create_agent
import gc, socket, asyncio

# Config flag for MCP parser usage
USE_MCP_PARSER = False  

datasets = [
    ("data/customs.gov.sg-Major Exporter Scheme (1).pdf", "singapore_tax"),
    ("data/cat-facts.txt", "cat_facts"),
    ("data/Interpretable Time-Series Few-shot.pdf", "DPSN_fewshotlearning_time_series"),
]

vectorstores = {}
weaviate_clients = []

async def async_load_all():
    tasks = [inject_dataset(fp, topic, use_mcp_parser=USE_MCP_PARSER) for fp, topic in datasets]
    results = await asyncio.gather(*tasks)
    for store_dict, client in results:
        vectorstores.update(store_dict)
        weaviate_clients.append(client)

asyncio.run(async_load_all())

tools = setup_tools_async(vectorstores, datasets)
agent = create_agent(tools)
llm = LangchainOllama(model="openhermes")

# Cleanup
def close_resources():
    try:
        for client in weaviate_clients:
            try:
                client.close()
                print("Weaviate client closed.")
            except Exception as e:
                print(f"Error closing Weaviate client: {e}")
    except Exception as e:
        print(f"Error during client loop: {e}")
    
    try:
        global llm
        if llm is not None:
            llm = None
        gc.collect()
        print("Ollama client dereferenced.")
    except Exception as e:
        print(f"Error during Ollama cleanup: {e}")

    try:
        import socket as socket_module
        for obj in gc.get_objects():
            if isinstance(obj, socket_module.socket):
                try:
                    if hasattr(obj, 'fileno') and obj.fileno() != -1:
                        obj.close()
                except (OSError, ValueError):
                    pass
                except Exception as e:
                    print(f"Error closing socket: {e}")
    except Exception as e:
        print(f"Error during socket cleanup: {e}")


def reload_datasets_with_parser(use_mcp: bool):
    """Reload all datasets with parser option."""
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.exceptions import WeaviateBaseError
    import os
    from dotenv import load_dotenv
    import threading
    import time
    
    global vectorstores, weaviate_clients, tools, agent
    
    load_dotenv()

    try:
        for client in weaviate_clients:
            try:
                client.close()
                print("Weaviate client closed.")
            except Exception as e:
                print(f"Error closing Weaviate client: {e}")
    except Exception as e:
        print(f"Error during client cleanup: {e}")
    
    # Clear existing data
    vectorstores = {}
    weaviate_clients = []

    time.sleep(1)
    
    # Delete existing collections before reloading
    temp_client = None
    try:
        temp_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        )
        
        if temp_client.is_ready():
            for _, topic in datasets:
                try:
                    temp_client.collections.delete(topic)
                    print(f"Deleted collection: {topic}")
                except WeaviateBaseError as e:
                    print(f"Collection {topic} not found: {e}")
                except Exception as e:
                    print(f"Error deleting collection {topic}: {e}")
    except Exception as e:
        print(f"Error connecting to Weaviate for cleanup: {e}")
    finally:
        if temp_client:
            try:
                temp_client.close()
            except Exception as e:
                print(f"Error closing temp client: {e}")

    time.sleep(2)


    def run_async_reload():
        # Create a new event loop for this thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        
        try:
            async def async_reload():
                tasks = [inject_dataset(fp, topic, use_mcp_parser=use_mcp) for fp, topic in datasets]
                results = await asyncio.gather(*tasks)
                for store_dict, client in results:
                    vectorstores.update(store_dict)
                    weaviate_clients.append(client)
            
            new_loop.run_until_complete(async_reload())
        finally:
            try:
                new_loop.close()
            except Exception as e:
                print(f"Error closing event loop: {e}")

    reload_thread = threading.Thread(target=run_async_reload)
    reload_thread.start()
    reload_thread.join()

    tools = setup_tools_async(vectorstores, datasets)
    agent = create_agent(tools)
    
    print(f"Datasets reloaded with {'MCP' if use_mcp else 'PyPDF2'} parser.")