from dotenv import load_dotenv
import os
import time
import PyPDF2
from docx import Document
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load environment variables
load_dotenv()

# Fetch API key and embedding model from .env
OPENAI_KEY = os.getenv("OPENAI_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Initialize ChromaDB client
chroma_client = chromadb.Client(base_url="http://localhost:8000")

# Set up OpenAI embedding function using the fetched variables
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name=EMBEDDING_MODEL)

# Access or create a collection
collection = chroma_client.get_collection(name="documents") or chroma_client.create_collection(name="documents", embedding_function=embedding_function)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        time.sleep(2)  # Wait for 2 seconds to ensure the file is fully written
        # Check if the modified file is a PDF or DOCX
        if event.src_path.endswith('.pdf') or event.src_path.endswith('.docx'):
            try:
                # Parse and index the document into ChromaDB
                document_text = None
                if event.src_path.endswith('.pdf'):
                    document_text = extract_text_from_pdf(event.src_path)
                elif event.src_path.endswith('.docx'):
                    document_text = extract_text_from_docx(event.src_path)
                
                if document_text:
                    # Generate embedding for the document
                    document_embedding = embedding_function.embed_text(document_text)
                    # Add the document, its embedding, and file path as metadata to ChromaDB
                    collection.add(documents=[document_text], embeddings=[document_embedding], metadatas=[{"file_path": event.src_path}])
            except Exception as e:
                print(f"Error processing file {event.src_path}: {e}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        try:
            # Search ChromaDB for the deleted file's path or name
            results = collection.query(query_texts=[event.src_path], where={"metadata": "file_path"})
            if results and 'documents' in results and results['documents']:
                # If found, remove the entry from ChromaDB
                document_id = results['documents'][0]['id']
                collection.remove(ids=[document_id])
                print(f"Removed entry for deleted file: {event.src_path}")
        except Exception as e:
            print(f"Error processing deletion of file {event.src_path}: {e}")

if __name__ == '__main__':
    directory_to_watch = 'chroma-openai\\bot_files'
    
    # Initialize the observer and event handler
    observer = Observer()
    event_handler = FileChangeHandler()
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)
    
    # Start the observer
    observer.start()
    
    try:
        while True:
            # Keep the script running
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
