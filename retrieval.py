import asyncio
import json
from typing import Dict
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.vectorstores.base import VectorStore
from langchain.tools.tavily_search import TavilySearchResults
from langchain.llms import Ollama as LangchainOllama
from langchain.schema import AgentAction, AgentFinish
from pdf_mcp_server.mcp_client import mcp_summarize_pdf, mcp_search_pdf
import requests

from pdf_mcp_server.mcp_client import mcp_tavily_search


FAILED_ACTIONS = []
MAX_FAILURES = 3  

class ForgivingReActParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        try:
            return super().parse(text)
        except Exception:
           
            if "Final Answer:" in text:
                final_text = text.split("Final Answer:")[-1].strip()
                return AgentFinish(return_values={"output": final_text}, log=text)

            if "Action:" in text and "Action Input:" in text:
                action = text.split("Action:")[1].split("\n")[0].strip()
                action_input = text.split("Action Input:")[1].split("\n")[0].strip().strip('"')

                if (action, action_input) not in FAILED_ACTIONS:
                    FAILED_ACTIONS.append((action, action_input))

                if len(FAILED_ACTIONS) >= MAX_FAILURES:
                    return AgentFinish(
                        return_values={"output": "ERROR — Too many failed tool attempts. Stopping."},
                        log=text
                    )

                return AgentAction(tool=action, tool_input=action_input, log=text)

            # Absolute last resort: return the whole text as the final output
            return AgentFinish(return_values={"output": text.strip()}, log=text)


def setup_tools_async(vectorstores: Dict[str, VectorStore], datasets):
    tools = []

    for topic, vectorstore in vectorstores.items():
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def make_retriever_fn(r):
            def _fn(q):
                if isinstance(q, dict):
                    q = " ".join(f"{k}: {v}" for k, v in q.items())
                elif isinstance(q, (list, tuple)):
                    q = " ".join(map(str, q))
                elif not isinstance(q, str):
                    q = str(q)
                return "\n".join(doc.page_content for doc in r.get_relevant_documents(q))
            return _fn

        retriever_tool = Tool.from_function(
            func=make_retriever_fn(retriever),
            name=f"{topic.capitalize()}Retriever",
            description=f"Use this to get information related to {topic.replace('-', ' ')}."
        )
        tools.append(retriever_tool)

    topic_to_file = {topic: fp for fp, topic in datasets}

    for topic in topic_to_file.keys():
        file_path = topic_to_file.get(topic)
        async def search_pdf_async(query: str, fp=file_path) -> str:
            result = await mcp_search_pdf(fp, query)
            return json.dumps(result, ensure_ascii=False)

        def search_pdf_sync(query: str, fp=file_path) -> str:
            return asyncio.run(search_pdf_async(query, fp))

        async def summarize_pdf_async(max_sentences: str, fp=file_path) -> str:
            try:
                max_sentences_int = int(max_sentences.strip())
            except ValueError:
                return "ERROR — Expected an integer for max_sentences"
            result = await mcp_summarize_pdf(fp, max_sentences_int)
            return json.dumps(result, ensure_ascii=False)

        def summarize_pdf_sync(max_sentences: str, fp=file_path) -> str:
            return asyncio.run(summarize_pdf_async(max_sentences, fp))
        
        tools.append(Tool.from_function(
            func=search_pdf_sync,
            name=f"{topic.capitalize()}SearchPDF",
            description=f"Search for a query inside the PDF for topic '{topic}'. Input: query string."
        ))
        tools.append(Tool.from_function(
            func=summarize_pdf_sync,
            name=f"{topic.capitalize()}SummarizePDF",
            description=f"Summarize the PDF for topic '{topic}'. Input: max sentences as integer."
        ))
        
    async def tavily_async(query: str) -> str:
        result = await mcp_tavily_search(query)
        return json.dumps(result, ensure_ascii=False)


    def tavily_sync(query: str) -> str:
        return asyncio.run(tavily_async(query))

    tavily_tool = Tool.from_function(
        func=tavily_sync,
        name="TavilySearch",
        description="Useful for searching current events and today's news."
    )
    tools.append(tavily_tool)

    return tools


def create_agent(tools):
    llm = LangchainOllama(model="openhermes", temperature=0.1, top_p=0.7)
    output_parser = ForgivingReActParser()

    failure_memory_text = ""
    if FAILED_ACTIONS:
        failure_memory_text = "\nYou already tried the following actions and they failed:\n"
        for tool, inp in FAILED_ACTIONS:
            failure_memory_text += f'- {tool}("{inp}")\n'
        failure_memory_text += "Do not attempt them again.\n"

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        agent_kwargs={
            "system_message": (
                failure_memory_text +
                "You are a highly disciplined and rule-bound assistant that MUST follow this format:\n"
                "Here is the conversation so far:\n{chat_history}\n\n"
                "Absolutely NO deviations, no summaries, and no creative liberties are allowed. This is MANDATORY:\n\n"
                "FORMAT:\n"
                "Thought: <your internal reasoning about the problem>\n"
                "Action: <one of the tool names exactly as given>\n"
                "Action Input: \"<input to the tool, as a string>\"\n\n"
                "After receiving the tool's output, continue with:\n"
                "Observation: <tool's output>\n"
                "Thought: <new reasoning>\n"
                "Action: ...\n"
                "OR\n"
                "Final Answer: <your final answer to the user>\n\n"
                "RULES:\n"
                "- You MUST use the tools provided when necessary.\n"
                "- Only call tools using the exact input schema they were defined with.\n"
                "- Only call tools with a single, concise string argument. Do not pass multiple arguments or objects.\n"
                "- If an answer is found based on the tools, MUST use that answer.\n"
                "- If tool usage is not needed, directly give a 'Final Answer'.\n"
                "- DO NOT return partial JSON, markdown, or chat-style messages.\n"
                "- DO NOT say 'Here's what I found' or make assumptions without using the tools.\n"
                "- If your action_input is null or empty, just copy the user's original question into it.\n"
                "- If you are unsure or confused, respond with:\n"
                "Final Answer: ERROR — Invalid input or tool failure.\n"
                "- If you violate the format, your response will be rejected.\n"
                "- Before interpreting vague references (e.g., 'it', 'that', 'they'), check your own chat_history to resolve what the pronoun specifically refers to. Do not interpret pronouns literally.\n"
                "- For example: If the user first says 'Vietnam’s GDP rose by 5.7%', and later says 'How will it affect trade?', you must understand that 'it' refers to the GDP increase."
                "Strictly obey this instruction at all times."
            )
        },
        output_parser=output_parser,
    )
