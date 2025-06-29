from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import List, Tuple
from pydantic import BaseModel


# import the .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

#initate the model
llm = ChatOpenAI(temperature= 0.5, model='gpt-4o-mini')

#connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

# Setup the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})


def sse_format(data: str) -> str:
    # Formats text as SSE message
    return f"data: {data}\n\n"

# call this function for every message added to the chatbot
def stream_response(message, history):
    # retriever the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"
    
    if message is not None:
        partial_message = ""

        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        # print(response.content)
        # partial_message = ""
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            # print(response.content, partial_message)
            # yield partial_message
            yield sse_format(response.content)
        
        # return partial_message

# # initiate the Gradio app
# chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
#                                                                container= False,
#                                                                autoscroll=True,
#                                                                scale = 7
#                                                                ))
# # launch the Gradio app
# chatbot.launch()


class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]] = []

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    generator = stream_response(req.message, req.history)
    return StreamingResponse(generator, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked"
    })