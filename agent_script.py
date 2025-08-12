from injection import inject_dataset
from retrieval import setup_tools_async, create_agent
from langchain.llms import Ollama as LangchainOllama
import gc, socket, asyncio

datasets = []

vectorstores = {}
weaviate_clients = []

async def async_load_all():
    tasks = [inject_dataset(fp, topic) for fp, topic in datasets]
    results = await asyncio.gather(*tasks)
    for store_dict, client in results:
        vectorstores.update(store_dict)
        weaviate_clients.append(client)

asyncio.run(async_load_all())

tools = setup_tools_async(vectorstores)
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
        llm = None
        gc.collect()
        print("Ollama client dereferenced.")
    except Exception as e:
        print(f"Error during Ollama cleanup: {e}")
    
    try:
        for obj in gc.get_objects():
            if isinstance(obj, socket.socket):
                try:
                    if obj.fileno() != -1:
                        obj.close()
                except:
                    pass
    except Exception as e:
        print(f"Error while closing lingering sockets: {e}")
