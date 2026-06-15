import streamlit as st

from services.pdf_service import extract_text_from_pdf
from services.chunking_service import create_chunks
from services.embedding_service import create_embeddings

st.title("AI Document Assistant")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    text = extract_text_from_pdf(uploaded_file)

    chunks = create_chunks(text)
    embeddings = create_embeddings(chunks)
    st.success("PDF uploaded successfully!")

    st.subheader("Extracted Text")

    st.text_area(
        "Document Content",
        text,
        height=300
    )

    st.write(f"Total Chunks: {len(chunks)}")

    with st.expander("View Chunks"):
        for index, chunk in enumerate(chunks):
            st.write(f"Chunk {index + 1}")
            st.text(chunk[:500])
    
    st.write(f"Total Embeddings: {len(embeddings)}")

    if len(embeddings) > 0:
        st.write(
            f"Embedding Dimensions: {len(embeddings[0])}"
        )        