import os
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOllama(model="mistral")
fast_llm = ChatOllama(model="mistral")

#llm = ChatOllama(model="llama3.1", temperature=0)


class RouterState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    category: str
    response: str


# --- Nodes ---

def classify_node(state: RouterState) -> dict:
    """Classify the incoming message."""
    last_message = state["messages"][-1].content

    response = fast_llm.invoke([
        SystemMessage(content=(
            "Classify this message into ONE category. "
            "Respond with ONLY the category name.\n"
            "Categories: question, complaint, feature_request, praise"
        )),
        HumanMessage(content=last_message)
    ])

    category = response.content.strip().lower()
    valid_categories = {"question", "complaint", "feature_request", "praise"}
    if category not in valid_categories:
        category = "question"

    print(f"[Classifier] Category: {category}")
    return {"category": category}


def handle_question(state: RouterState) -> dict:
    """Handle a question with a helpful answer."""
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the question clearly and concisely."),
        *state["messages"]
    ])
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def handle_complaint(state: RouterState) -> dict:
    """Handle a complaint with empathy and a solution."""
    response = llm.invoke([
        SystemMessage(content=(
            "You are a customer service agent. The user has a complaint. "
            "Respond with empathy, acknowledge the issue, and offer a solution or next steps."
        )),
        *state["messages"]
    ])
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def handle_feature_request(state: RouterState) -> dict:
    """Handle a feature request by acknowledging and logging it."""
    response = llm.invoke([
        SystemMessage(content=(
            "You are a product manager assistant. The user is requesting a feature. "
            "Thank them for the suggestion, ask clarifying questions if needed, "
            "and let them know the feedback has been logged."
        )),
        *state["messages"]
    ])
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def handle_praise(state: RouterState) -> dict:
    """Handle positive feedback graciously."""
    response = llm.invoke([
        SystemMessage(content=(
            "The user is giving positive feedback. Thank them warmly "
            "and let them know their feedback motivates the team."
        )),
        *state["messages"]
    ])
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


# --- Routing ---

def route_by_category(state: RouterState) -> Literal[
    "handle_question", "handle_complaint",
    "handle_feature_request", "handle_praise"
]:
    """Route to the appropriate handler based on classification."""
    category = state.get("category", "question")
    return f"handle_{category}"


# --- Build Graph ---

workflow = StateGraph(RouterState)

workflow.add_node("classify", classify_node)
workflow.add_node("handle_question", handle_question)
workflow.add_node("handle_complaint", handle_complaint)
workflow.add_node("handle_feature_request", handle_feature_request)
workflow.add_node("handle_praise", handle_praise)

workflow.add_edge(START, "classify")

workflow.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "handle_question": "handle_question",
        "handle_complaint": "handle_complaint",
        "handle_feature_request": "handle_feature_request",
        "handle_praise": "handle_praise",
    }
)

workflow.add_edge("handle_question", END)
workflow.add_edge("handle_complaint", END)
workflow.add_edge("handle_feature_request", END)
workflow.add_edge("handle_praise", END)

router_agent = workflow.compile()


# --- Test It ---

def process_message(message: str) -> str:
    result = router_agent.invoke({
        "messages": [HumanMessage(content=message)],
        "category": "",
        "response": "",
    })
    return result["response"]


if __name__ == "__main__":
    test_messages = [
        "How do I reset my password?",
        "Your app crashed three times today and I lost my work!",
        "It would be great if you added dark mode.",
        "I love this product! Best tool I've used.",
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = process_message(msg)
        print(f"Agent: {response}")
        print("-" * 60)
