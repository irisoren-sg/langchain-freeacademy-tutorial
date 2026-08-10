from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from ddgs import DDGS
import os
from datetime import datetime

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

@tool
def save_report(question: str, summary_text: str) -> str:
    """Create a report in Markdown that includes the original user question and the summary text
    of the search results and save it to the local folder.

    Args:
      question: The original text that was submitted by the user as a question.
      summary_text: The text that has been summarised.

    Returns:
      A message telling the caller where the file was saved.
    """
    try:
        # Where to store reports (relative to where you run the script)
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)

        # Use a timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_question_snippet = "".join(
            ch for ch in question.strip()[:40] if ch.isalnum() or ch in (" ", "-", "_")
        ).strip().replace(" ", "_")
        filename = f"report_{timestamp}_{safe_question_snippet or 'question'}.md"
        filepath = os.path.join(reports_dir, filename)

        report_md = f"""# Research Report

**Question:**
{question}

**Summary:**
{summary_text}

---

Saved at: {datetime.now().isoformat()}
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_md)

        return f"Report saved to: {filepath}"

    except Exception as e:
        return f"Failed to save report: {str(e)}"
      
# Create the agent — points at local Ollama server instead of OpenAI
llm = ChatOllama(model="llama3.1", temperature=0)
tools = [web_search, summarize_text, save_report]

agent = create_agent(
    model=llm,
    tools=tools,
)

system_message = SystemMessage(
    content=(
        "You are a research assistant. For every question, you MUST complete "
        "these steps in order, using tools:\n"
        "1. Call web_search to find relevant, current information.\n"
        "2. Call summarize_text to condense the results.\n"
        "3. Call save_report with the original question and your summary. "
        "This step is REQUIRED — do not skip it, and do not produce your "
        "final answer until save_report has been called.\n"
        "Only after save_report succeeds should you give your final answer, "
        "including the file path it returned and citing your sources."
    )
)

def research(question: str) -> str:
    print(f"\nResearching: {question}\n")
    result = agent.invoke({
        "messages": [system_message, HumanMessage(content=question)]
    })

    # Debug: show which tools were actually invoked
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[tool called] {tc['name']} args={tc['args']}")

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
