from langchain_ollama.chat_models import ChatOllama


##############################################
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



model = ChatOllama(model = "llama3", 
                   temperature = 0)



prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

chain = prompt | model

# Store for conversation histories (keyed by session ID)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# First message
config = {"configurable": {"session_id": "user-123"}}

response1 = chain_with_history.invoke(
    {"input": "My name is Alice and I love hiking."},
    config=config
)
print(response1.content)
# "Nice to meet you, Alice! Hiking is a wonderful hobby..."

# Second message — the agent remembers
response2 = chain_with_history.invoke(
    {"input": "What outdoor activities would you recommend for me?"},
    config=config
)
print(response2.content)