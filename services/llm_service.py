import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    token=os.getenv("HF_TOKEN")
)


def generate_answer(
    question,
    context
):
    try:

        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a document question answering assistant.

Answer ONLY using the provided context.

Rules:
- Give concise answers.
- Use bullet points for lists.
- Do not invent information.
- If the answer is not found in the context, say:

I could not find the answer in the document.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}
"""
                }
            ],
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as error:

        return f"Error generating answer: {error}"