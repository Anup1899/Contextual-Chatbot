from langchain_community.document_loaders import PyPDFDirectoryLoader   
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
# from chromadb import Sett
# from chromadb import Client
from uuid import uuid4


# Import the .env file
from dotenv import load_dotenv

load_dotenv()

# Configuration 
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")


# Set up Chroma with telemetry disabled
# settings = __settings(telemetry=False)
#initate the vector store
vector_store = Chroma(
    collection_name= "example_collection",
    embedding_function= embeddings_model,
    persist_directory=CHROMA_PATH,
    # client=Client(settings)
)

#Step 1 :- Load the PDF Documents
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

#Step 2 :- Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    length_function = len,
    is_separator_regex= False
)

# Creating the Chunks
chunks = text_splitter.split_documents(raw_documents)

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

#Adding Chinks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)