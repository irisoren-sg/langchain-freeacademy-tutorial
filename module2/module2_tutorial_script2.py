from langchain_ollama.chat_models import ChatOllama


##############################################
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define each piece
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical writer."),
    ("user", "Explain {topic} in 3 bullet points.")
])

model = ChatOllama(model="llama3", temperature=0)

output_parser = StrOutputParser()

# Compose them into a chain using the pipe operator
chain = prompt | model | output_parser

# Run the chain
result = chain.invoke({"topic": "machine learning"})
print(result)

