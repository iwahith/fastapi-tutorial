from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

openai_api_key = ""
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=100)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

UPLOAD_FOLDER = "uploads"  # Define your upload folder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

uploaded_files = []
pdf_qa = None

# Function to save uploaded file
def save_uploaded_file(file: UploadFile, upload_folder: str):
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path

# Upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = save_uploaded_file(file, UPLOAD_FOLDER)
        uploaded_files.append(file_path)

        # Creating Knowledge Base
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        global pdf_qa
        pdf_qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=retriever,
                                             memory=memory)
        print("Created Knowledge Based & Chain")
    
        return JSONResponse(content={"file_path": file_path,
                "status": "Created Knowledge Based & Chain"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: int):
        del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: int):
        await self.active_connections[client_id].send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            response = generate_response(data)  # Replace generate_response with your response generating function
            await manager.send_personal_message(response, client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

def generate_response(message: str) -> str:
    res = pdf_qa.invoke({"question": message})['answer']
    return res
