from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

def get_answer(query):
    print("Processing query:", query)

    # FREE embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load vector DB
    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    # Retrieve context
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    # FREE LLM
    llm = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question: {query}
    """

    response = llm(prompt, max_length=256, do_sample=True)

    return response[0]["generated_text"]

if __name__ == "__main__":
    while True:
        q = input("Ask: ")

        if q == "exit":
            break

        print(get_answer(q))