import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------------------------
# Pydantic model (documentation + type reference)
# --------------------------------------------------------------------

class CodeReview(BaseModel):
    language: str = Field(description="The programming language of the code")
    issues: list[str]
    suggestions: list[str]
    rating: int
    summary: str


# --------------------------------------------------------------------
# Prompt — NOTE: example JSON, NOT schema
# --------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert code reviewer.\n\n"
        "Respond ONLY with valid JSON.\n"
        "Fill in the values.\n\n"
        "Example response format:\n"
        "{{\n"
        '  "language": "python",\n'
        '  "issues": ["Issue 1", "Issue 2"],\n'
        '  "suggestions": ["Suggestion 1"],\n'
        '  "rating": 7,\n'
        '  "summary": "Bried (1-2 sentences)."\n'
        "}}\n\n"
        "Do not include explanations, markdown, or extra text."
    ),
    (
        "user",
        "Please review this code:\n\n"
        "```{language}\n{code}\n```"
    )
])

# --------------------------------------------------------------------
# Model
# --------------------------------------------------------------------

model = ChatOllama(
    model="llama3",
    temperature=0, 
    format = "json"  
    
)

chain = prompt | model

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Extremely robust JSON extraction for local LLM output.
    Handles invisible Unicode chars, markdown fences, and pre/post text.
    """
    if not text or not isinstance(text, str):
        raise RuntimeError("Empty or non-text response from model")

    # Normalize and strip obvious noise
    cleaned = text.strip()

    # 1. Fast path: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Brace slicing (most reliable for LLMs)
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_candidate = cleaned[start:end + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Found JSON-like content but failed to parse it.\n\n"
                f"Extracted text:\n{json_candidate}"
            ) from e

    # 3. Absolute failure
    raise RuntimeError(
        "No JSON object found in model response.\n\n"
        f"Raw response (repr):\n{repr(text)}"
    )


# --------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------

def review_code(code: str, language: str = "python") -> dict:
    response = chain.invoke({
        "code": code,
        "language": language
    })

    raw_text = response.content.strip()
    
    
    
    result = extract_json(raw_text)

    return result


def display_review(review: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"CODE REVIEW — {review['language'].upper()}")
    print(f"Rating: {review['rating']}/10")
    print(f"{'=' * 60}")

    print(f"\nSummary:\n{review['summary']}")

    if review["issues"]:
        print(f"\nIssues Found ({len(review['issues'])}):")
        for i, issue in enumerate(review["issues"], 1):
            print(f"  {i}. {issue}")

    if review["suggestions"]:
        print(f"\nSuggestions ({len(review['suggestions'])}):")
        for i, suggestion in enumerate(review["suggestions"], 1):
            print(f"  {i}. {suggestion}")

    print(f"\n{'=' * 60}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    sample_code = """
def get_user_data(id):
    import requests
    r = requests.get(f"http://api.example.com/users/{id}")
    data = r.json()
    name = data['name']
    email = data['email']
    age = data['age']
    return {"name": name, "email": email, "age": age, "raw": data}
"""

    print("Reviewing sample code...")
    review = review_code(sample_code, "python")
    display_review(review)

    print("\nPaste your own code to review (type 'done' on a new line to submit):")
    lines = []
    while True:
        line = input()
        if line.strip().lower() == "done":
            break
        lines.append(line)

    if lines:
        user_code = "\n".join(lines)
        language = input(
            "\nEnter the programming language (e.g. python, javascript, java): "
        ).strip().lower() or "python"

        print(f"\nReviewing your {language} code...")
        review = review_code(user_code, language)
        display_review(review)


if __name__ == "__main__":
    main()