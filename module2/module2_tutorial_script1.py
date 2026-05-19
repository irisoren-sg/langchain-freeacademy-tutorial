from langchain_ollama.chat_models import ChatOllama

llm_llama = ChatOllama(
    model="llama3",     # or llama2, mistral, etc.
    temperature=0
)

# Same interface, different providers
#response = llm_llama.invoke("What is the capital of France?")
#print(response.content)  # "The capital of France is Paris."

##############################################
from langchain_core.prompts import ChatPromptTemplate

# Define a reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Respond in {language}."),
    ("user", "{question}")
])

# Use it with different parameters
messages = prompt.invoke({
    "role": "helpful travel advisor",
    "language": "English",
    "question": "What should I visit in Tokyo?"
})

response = llm_llama.invoke(messages)
print(response.content)
