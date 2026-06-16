# YouTube Transcript RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to chat with YouTube videos using transcript-based retrieval. The system extracts transcripts from YouTube videos, creates vector embeddings using FAISS, retrieves relevant chunks, and generates answers using a Hugging Face LLM.

## Features

* Extract transcripts from YouTube videos
* Split transcripts into chunks
* Generate embeddings using Sentence Transformers
* Store vectors in FAISS
* Retrieve relevant transcript chunks
* Answer questions using a Hugging Face LLM
* Streamlit-based web interface

## Tech Stack

* Python
* LangChain
* FAISS
* Hugging Face Embeddings
* Hugging Face Inference API
* YouTube Transcript API
* Streamlit

## Application Screenshot

<img width="877" height="635" alt="Screenshot 2026-06-16 130637" src="https://github.com/user-attachments/assets/7f6a9f08-f99d-4bb2-a8be-7605074d0741" />


## Project Workflow

1. Extract transcript from a YouTube video.
2. Split transcript into chunks using RecursiveCharacterTextSplitter.
3. Generate embeddings using Sentence Transformers.
4. Store embeddings in FAISS.
5. Retrieve the most relevant chunks.
6. Send retrieved context to the LLM.
7. Generate answers grounded in the video transcript.

## Project Structure

```text
YouTube-Transcript-RAG-Chatbot/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .env
│
└── images/
    └── app_screenshot.png
```

## Installation

```bash
git clone https://github.com/your-username/YouTube-Transcript-RAG-Chatbot.git

cd YouTube-Transcript-RAG-Chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```

## Run the Application

```bash
streamlit run streamlit_app.py
```

## Example Questions

* Can you summarize the video?
* What are the key points discussed?
* Is AI discussed in the video?
* What examples were mentioned by the speaker?
* What conclusions were presented?

## Requirements

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-huggingface
faiss-cpu
sentence-transformers
youtube-transcript-api
python-dotenv
huggingface_hub
```

## Future Improvements

* Support multiple YouTube videos
* Add chat history memory
* Show retrieved source chunks
* Support multilingual transcripts
* Deploy using Streamlit Cloud

## License

This project is licensed under the MIT License.
