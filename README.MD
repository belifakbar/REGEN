**README File Summary 📄**

Welcome to our Readme-Generator project! 🔧 This comprehensive tool is designed to generate README files for GitHub repositories, leveraging AI-powered text generation capabilities 🔩.

<img src="./assets/readme-generatorrr.gif" alt="readme generatorrr" width="400"/>


**Overview 🌐**

Our project utilizes a Python script (`main.py`) that imports essential libraries and custom functions to work with Git repositories and generate documentation. The code initializes an Ollama model and sets up an environment using environment variables. An interactive AI agent is created, allowing for querying and response generation 💻.

**Features 🎉**

* **Tools 🔧**: A collection of reusable functions for processing GitHub documents:
	+ `get_git_docs`: Retrieves GitHub documents for a given repository.
	+ `code_reader_func`: Reads the contents of code files and returns them as a dictionary.
	+ `get_git_files`: Filters GitHub files by type (e.g., Python) and removes files containing certain keywords (e.g., "__").
	+ `readme_generator_func`: Generates a README file based on the given repository and input file.
* **GitDocs 📊**: A class responsible for managing the cache of GitHub documents, ensuring that documents are loaded only once and then cached for future use.
* **LLM (Large Language Model) Integration 💬**: Our code uses an Ollama LLM to generate text based on input prompts, enabling the generation of README files with accurate descriptions using natural language processing techniques.
* **Agent 🤖**: An interactive AI agent is created, configured with `tools.py` functions and the Ollama LLM. This allows for querying and response generation.

**How it Works 🔩**

1. The script enters an infinite loop where it:
	* Asks the user to input a prompt (or "q" to quit).
	* Tries to generate a response using the `ReActAgent` with the provided prompt.
	* If successful, prints the generated result and breaks out of the inner loop.
	* If not successful (after 3 retries), continues to the next iteration.
2. After processing each prompt, the script resets the `GitDocs._git_docs` variable to None, clearing any cached or previously generated documentation.

**Getting Started 🚀**

To use our Readme-Generator project, simply run the `main.py` file and follow the prompts. Our tool is designed to be easy to use and provides a comprehensive solution for generating README files for GitHub repositories.

**Acknowledgments 💡**

We would like to acknowledge the contributions of the Ollama team and their work on the LLM model. Their efforts have enabled us to integrate AI-powered text generation capabilities into our project.

**Contact 📲**

If you have any questions or feedback, please don't hesitate to reach out. We're always happy to help and improve our tool.

**License 🔒**

Our Readme-Generator project is licensed under the MIT License. See LICENSE.txt for more information.


---
**Demo 🖥️**
Section below is not written by AI.
To get started

1) Clone this repo
2) Install the requirements.txt
3) Run ollama using https://ollama.com. Alternatively, you may run colab_ollama on colab and use the free GPU.
3) Create a .env file with your GitHub token
4) Run the main.py file
5) Enter the GitHub repository URL!


```bash
git clone https://github.com/ivanleech/readme-generator.git
cd readme-generator
pip install -r requirements.txt
ollama serve & ollama run llama3:instruct # Separate terminal
python main.py
```

<img src="./assets/readme-generator.gif" alt="readme generator" width="800"/>
