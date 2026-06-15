import streamlit as st

from services.pdf_service import extract_text_from_pdf
from services.chunking_service import create_chunks
from services.embedding_service import (
    create_embeddings,
    create_embedding
)
from services.chroma_services import (
    clear_collection,
    store_chunks,
    search_chunks
)
from services.llm_service import (
    generate_answer
)

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title(
    "🤖 AI Document Assistant"
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    text = extract_text_from_pdf(
        uploaded_file
    )

    chunks = create_chunks(
        text
    )

    embeddings = create_embeddings(
        chunks
    )

    clear_collection()

    store_chunks(
        chunks,
        embeddings
    )

    st.success(
        "PDF uploaded successfully!"
    )

    st.write(
        f"📄 Source File: {uploaded_file.name}"
    )

    st.subheader(
        "Extracted Text"
    )

    st.text_area(
        "Document Content",
        text,
        height=300
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Chunks",
            len(chunks)
        )

    with col2:

        st.metric(
            "Embedding Dimensions",
            len(embeddings[0])
        )

    with st.expander(
        "View Chunks"
    ):

        for index, chunk in enumerate(
            chunks
        ):

            st.markdown(
                f"### Chunk {index + 1}"
            )

            st.write(
                chunk
            )

    st.divider()

    st.subheader(
        "Ask Questions About Your Document"
    )

    question = st.text_input(
        "Ask a question about the document"
    )

    if question:

        question_embedding = create_embedding(
            question
        )

        results = search_chunks(
            question_embedding,
            top_k=5
        )

        retrieved_chunks = results[
            "documents"
        ][0]

        context = "\n\n".join(
            retrieved_chunks
        )

        answer = generate_answer(
            question,
            context
        )

        st.subheader(
            "AI Answer"
        )

        st.success(
            answer
        )

        st.subheader(
            "Top Matching Chunks"
        )

        for index, chunk in enumerate(
            retrieved_chunks
        ):

            st.write(
                f"### Rank {index + 1}"
            )

            st.info(
                chunk
            )