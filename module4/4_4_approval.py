from typing import TypedDict, Annotated, Literal
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOllama(model="mistral")


class ApprovalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    draft_email: str
    approved: bool
    sent: bool


def draft_email_node(state: ApprovalState) -> dict:
    """Draft an email based on the user's request."""
    response = llm.invoke([
        SystemMessage(content=(
            "Draft a professional email based on the user's request. "
            "Return ONLY the email content (subject and body)."
        )),
        *state["messages"]
    ])

    draft = response.content
    print(f"\n--- DRAFT EMAIL ---\n{draft}\n-------------------\n")
    return {"draft_email": draft}


def review_node(state: ApprovalState) -> dict:
    """Present the draft for human review."""
    # This node uses an interrupt to pause execution
    # In a real app, this would present a UI for approval
    print("\nThis email requires your approval before sending.")
    print(f"\nDraft:\n{state['draft_email']}\n")

    approval = input("Approve this email? (yes/no): ").strip().lower()
    approved = approval == "yes"

    return {"approved": approved}


def send_email_node(state: ApprovalState) -> dict:
    """Send the approved email."""
    print("[Email] Sending email...")
    # In production, this would call an email API
    return {
        "sent": True,
        "messages": [AIMessage(content="Email sent successfully.")]
    }


def reject_node(state: ApprovalState) -> dict:
    """Handle rejected email."""
    return {
        "sent": False,
        "messages": [AIMessage(content="Email was not sent. Let me know if you'd like to revise it.")]
    }


def route_after_review(state: ApprovalState) -> Literal["send", "reject"]:
    """Route based on approval status."""
    if state.get("approved", False):
        return "send"
    return "reject"


# Build the graph
workflow = StateGraph(ApprovalState)

workflow.add_node("draft", draft_email_node)
workflow.add_node("review", review_node)
workflow.add_node("send", send_email_node)
workflow.add_node("reject", reject_node)

workflow.add_edge(START, "draft")
workflow.add_edge("draft", "review")

workflow.add_conditional_edges(
    "review",
    route_after_review,
    {"send": "send", "reject": "reject"}
)

workflow.add_edge("send", END)
workflow.add_edge("reject", "draft")

# Compile with a checkpointer for state persistence
checkpointer = MemorySaver()
# Compile with interrupt_before — pauses before the "send" node
email_agent = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["send"]
)

# First invocation — runs until it hits the interrupt
config = {"configurable": {"thread_id": "email-1"}}
result = email_agent.invoke(
    {
        "messages": [HumanMessage(content="Send a follow-up email to the client about the project deadline")],
        "draft_email": "",
        "approved": False,
        "sent": False,
    },
    config=config,
)

# The agent paused before "send" — review the draft
print(f"Draft: {result['draft_email']}")

# To approve and continue:
email_agent.invoke(None, config=config)  # Resumes from checkpoint

# To reject: update state and re-invoke with a different path
