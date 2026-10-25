
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def generate_answer_from_documents(question, documents):

    if not documents:
        return "No relevant information found in the uploaded PDF."

    context = ""

    for i, doc in enumerate(documents, start=1):
        context += f"\nSource {i}:\n"
        context += doc.page_content[:1500]
        context += f"\nPage: {doc.metadata.get('page', 'Unknown')}\n"

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        timeout=120,
        max_retries=0
    )

    prompt = f"""
You are an AI Research Assistant.

Answer the question using only the document content provided below.

If the answer is not available, clearly mention that
the document does not provide enough information.

Document Content:
{context}

Question:
{question}

Provide a clear and well-organized answer.
"""

    response = None

    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            break

        except Exception as e:
            error_message = str(e)

            if "503" in error_message or "504" in error_message:
                if attempt < 2:
                    wait_time = 5 * (attempt + 1)
                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content)
