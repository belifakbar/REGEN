import nest_asyncio

from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from tools import get_git_files, readme_generator, GitDocs, summarize
from prompts import context
import os

nest_asyncio.apply()

base_url = os.environ["OLLAMA_HOST"]
llm = Ollama(model="llama3:instruct", base_url=base_url, request_timeout=300)
tools = [get_git_files, readme_generator, summarize]

agent = ReActAgent.from_tools(
    tools, llm=llm, verbose=True, context=context, max_iterations=30
)


while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0
    while retries < 3:
        try:
            result = agent.query(prompt)
            print("result", result)
            break
        except Exception as e:
            retries += 1
            print(f"exception occurered, retry #{retries} - {e}")

    if retries >= 3:
        continue

    GitDocs._git_docs = None
