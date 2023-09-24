# main.py

from dotenv import load_dotenv
import os
import openai
import pprint
from halo import Halo
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import PyPDF2
from docx import Document
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
pp = pprint.PrettyPrinter(indent=4)

# Initialize OpenAIEmbeddingFunction
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_KEY"),
    model_name=os.getenv("EMBEDDING_MODEL")
)

def generate_response(messages):
    spinner = Halo(text='Loading...', spinner='dots')
    spinner.start()
    openai.api_key = os.getenv("OPENAI_KEY")
    model_name = os.getenv("MODEL_NAME")
    response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=250)

    spinner.stop()
    print("Request:")
    pp.pprint(messages)

    print(f"Completion tokens: {response['usage']['completion_tokens']}, Prompt tokens: {response['usage']['prompt_tokens']}, Total tokens: {response['usage']['total_tokens']}")
    return response['choices'][0]['message']

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
    def __init__(self, embedding_function, collection):
        self.embedding_function = embedding_function
        self.collection = collection

    def on_modified(self, event):
        if event.is_directory:
            return
        time.sleep(2)  # Wait for 2 seconds to ensure the file is fully written
        if event.src_path.endswith('.pdf') or event.src_path.endswith('.docx'):
            try:
                document_text = None
                if event.src_path.endswith('.pdf'):
                    document_text = extract_text_from_pdf(event.src_path)
                elif event.src_path.endswith('.docx'):
                    document_text = extract_text_from_docx(event.src_path)
                
                if document_text:
                    logger.info(f"Fetching information from {event.src_path}")
                    document_embedding = self.embedding_function.embed_text(document_text)
                    self.collection.add(documents=[document_text], embeddings=[document_embedding], metadatas=[{"file_path": event.src_path}])
            except Exception as e:
                logger.error(f"Error processing file {event.src_path}: {e}")
                print(f"Error processing file {event.src_path}: {e}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        try:
            results = self.collection.query(query_texts=[event.src_path], where={"metadata": "file_path"})
            if results and 'documents' in results and results['documents']:
                document_id = results['documents'][0]['id']
                self.collection.remove(ids=[document_id])
                print(f"Removed entry for deleted file: {event.src_path}")
        except Exception as e:
            print(f"Error processing deletion of file {event.src_path}: {e}")

def start_file_watcher(embedding_function, collection):
    directory_to_watch = os.path.join(os.getcwd(), 'bot_files')
    observer = Observer()
    event_handler = FileChangeHandler(embedding_function, collection)
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except:
        observer.stop()
    observer.join()

def main():
    chroma_client = chromadb.Client()
    embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_KEY"), model_name=os.getenv("EMBEDDING_MODEL"))
    collection = chroma_client.create_collection(name="conversations", embedding_function=embedding_function)

    # Start the file watcher in a separate thread
    file_watcher_thread = threading.Thread(target=start_file_watcher, args=(embedding_function, collection))
    file_watcher_thread.start()

    current_id = 0  # Initialize current_id here

    while True:
        chat_history = []
        chat_metadata = []
        history_ids = []

        messages=[
            {"role": "system", "content": "You are a kind and wise wizard"}
            ]
        input_text = input("You: ")
        if input_text.lower() == "quit":
            break

        results = collection.query(
            query_texts=[input_text],
            where={"role": "assistant"},
            n_results=2
        )

        # Log when fetching from documents
        if results and results.get('documents'):
            logger.info(f"Fetched information from documents for query: {input_text}")

        # append the query result into the messages
        for res in results['documents'][0]:
            messages.append({"role": "user", "content": f"previous chat: {res}"})

        # append user input at the end of conversation chain
        messages.append({"role": "user", "content": input_text})
        response = generate_response(messages)

        chat_metadata.append({"role":"user"})
        chat_history.append(input_text)
        chat_metadata.append({"role":"assistant"})
        chat_history.append(response['content'])
        current_id += 1
        history_ids.append(f"id_{current_id}")
        current_id += 1
        history_ids.append(f"id_{current_id}")
        collection.add(
            documents=chat_history,
            metadatas=chat_metadata,
            ids=history_ids
        )
        print(f"Wizard: {response['content']}")

    # Optionally, if you want to ensure the file watcher stops when your main application stops:
    file_watcher_thread.join()

if __name__ == "__main__":
    main()

