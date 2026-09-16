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
You are a question answering assistant.

Use ONLY the provided context to answer the question.

Return ONLY the short answer.
Do not explain your reasoning.
Do not add any extra text.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200
    )

    return response.choices[0].message.content
