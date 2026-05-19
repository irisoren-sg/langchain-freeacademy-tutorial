This repo contains code from
 https://freeacademy.ai/courses/agentic-ai-python-langchain 
that has been modified to run with llama3

Setup  instructions 

1. Install ollama and llama3

```
# Install Ollama: https://ollama.com
# Then pull Llama3:
ollama pull llama3

# Run a quick interactive test:
ollama run llama3
```

2. Create python environment

```
python -m venv agent_env
# macOS/Linux:
source agent_env/bin/activate
# Windows:
# agent_env\Scripts\activate
```

3. Install LangChain  + LangGraph + Ollama bindings

```
pip install langchain langgraph langchain-community ollama langchain-ollama
```


