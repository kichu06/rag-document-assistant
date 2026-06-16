import os
import streamlit as st

from dotenv import load_dotenv
from huggingface_hub import (
    InferenceClient
)

load_dotenv()

# Try Streamlit Secrets first
try:
    token = st.secrets["HF_TOKEN"]

except Exception:
    token = os.getenv(
        "HF_TOKEN"
    )

client = InferenceClient(
    token=token
)


def generate_answer(
    question,
    context
):

    prompt = f"""
You are a document question answering assistant.

Use ONLY the provided context.

If the answer is not present in the context,
reply exactly:

I could not find the answer in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200
    )

    return response.choices[0].message.content