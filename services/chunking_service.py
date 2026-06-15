import re


def create_chunks(
    text,
    chunk_size=5,
    overlap=2
):
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    chunks = []

    step = chunk_size - overlap

    for index in range(
        0,
        len(sentences),
        step
    ):
        chunk = " ".join(
            sentences[
                index:index + chunk_size
            ]
        )

        if len(chunk.split()) >= 15:
            chunks.append(chunk)

    return chunks