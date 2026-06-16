
# ! pip install -q youtube-transcript-api

from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

#indexing

video_id="-HzgcbRXUK8"
yt_api = YouTubeTranscriptApi()
try:
  transcript_list=yt_api.fetch(video_id,languages=["en"])
  transcript=" ".join(chunk.text for chunk in transcript_list)
  print(transcript)

except:
  print("Transcript not available for this video")

#text splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )
chunks=splitter.create_documents([transcript])

len(chunks)

#embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store=FAISS.from_documents(chunks,embeddings)

from re import search
#retrieval
retrieval=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)
retrieval.invoke('what is AI')

#Augmentation
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="""
      "you are a helpful assistant.
    answer only from the provided transcript context.
    if the context is insufficient , just say you don't know.
    {context}
    question:{question}
    """,


    input_variables=['context','question']
)

question="is the topic of alien discussed in this video? if yes then what was discussed"
retrieved_docs=retrieval.invoke(question)

retrieved_docs

context_text="\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt= prompt.invoke({'context':context_text,'question':question})
final_prompt

#Generation
answer=model.invoke(final_prompt)
print(answer.content)

def format_docs(retrieved_docs):
  context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain=RunnableParallel({
    'context':retrieval | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})

parser=StrOutputParser()

main_chain= parallel_chain | prompt | model | parser

print(main_chain.invoke('can you summarize the video'))

