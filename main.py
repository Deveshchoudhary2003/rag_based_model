import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)
from langchain_core.output_parsers import StrOutputParser

INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

rag_chain_cache = {}

# -------------------------------
# LAZY LOADERS (VERY IMPORTANT)
# -------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

def get_llm():
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        temperature=0.3
        )
    )

# -------------------------------
# TRANSCRIPT FETCH
# -------------------------------

def fetch_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript_list = transcript.to_raw_data()
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        raise ValueError("No captions available")

# -------------------------------
# FAISS INDEX
# -------------------------------

def build_and_save_index(video_id: str):
    path = f"{INDEX_DIR}/{video_id}"
    if os.path.exists(path):
        return

    transcript = fetch_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(docs, get_embeddings())
    vector_store.save_local(path)

def load_vector_store(video_id: str):
    return FAISS.load_local(
        f"{INDEX_DIR}/{video_id}",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

# -------------------------------
# RAG CHAIN
# -------------------------------

def get_rag_chain(video_id: str):
    if video_id in rag_chain_cache:
        return rag_chain_cache[video_id]

    build_and_save_index(video_id)

    retriever = load_vector_store(video_id).as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate(
        template="""
Answer only using the context.
If unknown, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(lambda d: d[0].page_content),
            "question": RunnablePassthrough()
        })
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    rag_chain_cache[video_id] = chain
    return chain

# -------------------------------
# PUBLIC FUNCTION (API CALLS THIS)
# -------------------------------

def ask_question(video_id: str, question: str) -> str:
    return get_rag_chain(video_id).invoke(question)


