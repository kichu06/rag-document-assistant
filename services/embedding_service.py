import streamlit as st

from sentence_transformers import (
    SentenceTransformer
)


@st.cache_resource
def load_model():

    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )


model = load_model()


def create_embeddings(
    chunks
):

    passages = [
        f"passage: {chunk}"
        for chunk in chunks
    ]

    return model.encode(
        passages
    )


def create_embedding(
    question
):

    return model.encode(
        f"query: {question}"
    )