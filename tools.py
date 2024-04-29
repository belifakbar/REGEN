from llama_index.core.tools import FunctionTool
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

import os

load_dotenv()
base_url = os.environ["OLLAMA_HOST"]
llm = Ollama(model="llama3:instruct", base_url=base_url, request_timeout=300)


class GitDocs:
    _git_docs = None

    # GITHUB documents
    @staticmethod
    def get_git_docs(
        owner: str = "ivanleech",
        repo: str = "rusty",
        branch: str = "main",
    ):
        if GitDocs._git_docs is None:
            github_token = os.environ["GITHUB_TOKEN"]
            owner = owner
            repo = repo
            branch = branch if branch != "" else "main"
            github_client = GithubClient(github_token)
            documents = GithubRepositoryReader(
                github_client,
                owner=owner,
                repo=repo,
                use_parser=False,
                verbose=False,
            ).load_data(branch=branch)
            GitDocs._git_docs = documents
        return GitDocs._git_docs


def code_reader_func(file_name):
    path = os.path.join(file_name)
    try:
        with open(path, "r") as f:
            content = f.read()
            return {"file_content": content}
    except Exception as e:
        return {"error": str(e)}


code_reader = FunctionTool.from_defaults(
    fn=code_reader_func,
    name="code_reader",
    description=" this tool read the contents of code files and return their results. Use this when you need to read contents of file",
)


def get_git_files_func(owner, repo, branch):
    documents = GitDocs.get_git_docs(owner, repo, branch)
    file_type = "py"  # file type to filter. Update according to repo
    git_files = [
        doc.metadata["file_path"]
        for doc in documents
        if doc.metadata["file_path"].endswith(file_type)
        and "__" not in doc.metadata["file_path"]
    ]
    return {"git_files": git_files}


get_git_files = FunctionTool.from_defaults(
    fn=get_git_files_func,
    name="get_git_files",
    description="This tool takes in the github url, retrieves code from repository and return list of files",
)


def readme_generator_func(input_repo_file):
    documents = GitDocs.get_git_docs()
    embed_model = resolve_embed_model("local:BAAI/bge-small-en")
    vector_index = VectorStoreIndex(documents, embed_model=embed_model)
    query_engine = vector_index.as_query_engine(llm=llm)
    resp = query_engine.query(
        f"Describe the code of {input_repo_file} in concise manner with emojis."
    )
    try:
        with open("./ai/AI-README.MD", "a") as f:
            f.write(f"\n\n## {input_repo_file}\n\n")
            f.write(resp.response)
            f.write("\n\n")
        print(f"readme generated for {input_repo_file}")
    except Exception as e:
        print("error generating readme", str(e))
    return {
        "input_repo_file": input_repo_file,
        "readme_description": resp.response,
    }


readme_generator = FunctionTool.from_defaults(
    fn=readme_generator_func,
    name="readme_generator",
    description=(
        "This tool takes in file as input and explains what the file does with emojis."
        "return {'input_repo_file': 'input_repo_file', 'readme_description': 'description'}"
    ),
)


def cleanup_func():
    GitDocs._git_docs = None

    reader = SimpleDirectoryReader(
        input_dir="./ai",
    )
    documents = reader.load_data()
    embed_model = resolve_embed_model("local:BAAI/bge-small-en")
    vector_index = VectorStoreIndex(documents, embed_model=embed_model)
    query_engine = vector_index.as_query_engine(llm=llm)
    resp = query_engine.query(
        "Give me summary in the format of a proper README file with emojis. Double check to ensure contents are correct."
    )
    print("cleanup done")
    with open("./ai/AI-README-CLEAN.MD", "w") as f:
        f.write(resp.response)


cleanup = FunctionTool.from_defaults(
    fn=cleanup_func,
    name="cleanup",
    description="No input required. This tool summarizes and writes to AI-README-CLEAN.MD",
)
