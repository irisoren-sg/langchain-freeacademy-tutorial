from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from ddgs import DDGS

# Define the search tool — no API key needed
@tool
def web_search(query: str) -> str:
    """Search the web for current information on any topic.

    Args:
        query: The search query to look up.
    """
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=5))

        if not raw_results:
            return "No results found for the query."

        results = []
        for item in raw_results:
            results.append(
                f"Title: {item.get('title', '')}\n"
                f"URL: {item.get('href', '')}\n"
                f"Content: {item.get('body', '')[:300]}\n"
            )
        return "\n---\n".join(results)

    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Summarize a long text into a shorter version.

    Args:
        text: The text to summarize.
        max_sentences: Maximum number of sentences in the summary.
    """
    llm = ChatOllama(model="llama3.1", temperature=0)
    response = llm.invoke(
        f"Summarize the following text in {max_sentences} sentences:\n\n{text}"
    )
    return response.content

# Create the agent — points at local Ollama server instead of OpenAI
llm = ChatOllama(model="llama3.1", temperature=0)
tools = [web_search, summarize_text]

agent = create_agent(
    model=llm,
    tools=tools,
)

system_message = SystemMessage(
    content=(
        "You are a research assistant. When asked a question:\n"
        "1. Search the web for relevant, current information\n"
        "2. Analyze and synthesize the results\n"
        "3. Provide a clear, well-structured answer with sources\n"
        "4. If the search results are insufficient, search again with different terms\n"
        "Always cite your sources."
    )
)

def research(question: str) -> str:
    print(f"\nResearching: {question}\n")
    result = agent.invoke({
        "messages": [system_message, HumanMessage(content=question)]
    })
    return result["messages"][-1].content

def main():
    print("Web Search Agent (local, no API keys)")
    print("Type your research question (or 'exit' to quit).\n")

    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        answer = research(question)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
