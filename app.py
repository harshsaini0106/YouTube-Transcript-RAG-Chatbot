import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="YouTube RAG Chatbot")
st.title("YouTube Transcript RAG Chatbot")

video_id = st.text_input("Enter YouTube Video ID")

question = st.text_input("Ask a Question")

if st.button("Get Answer"):

    try:
        yt_api = YouTubeTranscriptApi()

        transcript_list = yt_api.fetch(
            video_id,
            languages=["en"]
        )

        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.create_documents([transcript])

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation"
        )

        model = ChatHuggingFace(llm=llm)

        prompt = PromptTemplate(
            template="""
You are a helpful assistant.

Answer only from the provided transcript context.

If the context is insufficient, say:
"I don't know based on the video transcript."

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"],
        )

        def format_docs(docs):
            return "\n\n".join(
                doc.page_content for doc in docs
            )

        chain = (
            RunnableParallel(
                {
                    "context": retriever
                    | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | model
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        st.success(answer)

    except Exception as e:
        st.error(f"Error: {e}")