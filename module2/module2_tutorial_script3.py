from langchain_ollama.chat_models import ChatOllama


##############################################
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

model = ChatOllama(model = "llama3", 
                   temperature = 0)

# Define the structure you want
class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="One-sentence summary")

parser = JsonOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the movie and respond in JSON format.\n{format_instructions}"),
    ("user", "Review the movie: {movie}")
])

chain = prompt | model | parser

result = chain.invoke({
    "movie": "The Matrix",
    "format_instructions": parser.get_format_instructions()
})

print(result)
