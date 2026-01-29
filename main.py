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

# -------------------------------
# GLOBAL OBJECTS (CACHED)
# -------------------------------

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

LLM = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        temperature=0.3
    )
)

rag_chain_cache = {}

INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# -------------------------------
# TRANSCRIPT FETCH
# -------------------------------

def fetch_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript_list = transcript.to_raw_data()
        return " ".join(chunk["text"] for chunk in transcript_list)

    except TranscriptsDisabled:
        raise ValueError("No captions available for this video")

    except Exception as e:
        raise ValueError(str(e))

# -------------------------------
# BUILD & SAVE FAISS (ONCE)
# -------------------------------

def build_and_save_index(video_id: str):
    index_path = f"{INDEX_DIR}/{video_id}"
    if os.path.exists(index_path):
        return  # already built

    transcript = fetch_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
    vector_store.save_local(index_path)

# -------------------------------
# LOAD FAISS (FAST)
# -------------------------------

def load_vector_store(video_id: str):
    return FAISS.load_local(
        f"{INDEX_DIR}/{video_id}",
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

# -------------------------------
# FORMAT DOCS (TOKEN CONTROL)
# -------------------------------

def format_docs(docs, max_chars=3000):
    text = ""
    for doc in docs:
        if len(text) + len(doc.page_content) > max_chars:
            break
        text += doc.page_content + "\n\n"
    return text

# -------------------------------
# BUILD RAG CHAIN (CACHED)
# -------------------------------

def get_rag_chain(video_id: str):
    if video_id in rag_chain_cache:
        return rag_chain_cache[video_id]

    build_and_save_index(video_id)

    vector_store = load_vector_store(video_id)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | LLM | StrOutputParser()

    rag_chain_cache[video_id] = main_chain
    return main_chain

# -------------------------------
# ASK QUESTION (FAST PATH)
# -------------------------------

def ask_question(video_id: str, question: str) -> str:
    chain = get_rag_chain(video_id)
    return chain.invoke(question)



