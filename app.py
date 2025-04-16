import os
import json
import hashlib
import time
from dotenv import load_dotenv
from document_processor import process_document_folder
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_openai import ChatOpenAI  # Ensure you have langchain-openai installed
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import gradio as gr
from langchain.embeddings.base import Embeddings
from langchain.schema import HumanMessage, AIMessage  # For role detection
from document_manager import DocumentManager
from upload_ui import initialize_document_manager, create_upload_tab


# Load environment variables from a .env file
load_dotenv()

# Retrieve configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
INDEXES_FOLDER = os.getenv("INDEXES_FOLDER", "indexes")

# --------------------
# Custom Embeddings Class implementing the Embeddings interface
# --------------------
class MyEmbeddings(Embeddings):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Instantiate our embeddings object
embeddings = MyEmbeddings()

# --------------------
# File Modification Check Functions
# --------------------
def load_file_modification_info(data_folder):
    mod_info = {}
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.docx', '.pdf', '.txt')):
                path = os.path.join(root, file)
                mod_info[path] = os.path.getmtime(path)
    return mod_info

# Helper functions for loading and saving the stored fingerprint (last_modified.json)
def load_previous_mod_info(mod_file_path):
    if os.path.exists(mod_file_path):
        with open(mod_file_path, 'r') as f:
            return json.load(f)  # expecting a simple hash string
    return None

def save_mod_info(mod_info, mod_file_path):
    with open(mod_file_path, 'w') as f:
        json.dump(mod_info, f)

# --------------------
# Index Building / Rebuilding Functions
# --------------------
def rebuild_index(data_folder, indexes_folder):
    print("[INFO] Rebuilding index from documents...")
    # Make sure the folders exist
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(indexes_folder, exist_ok=True)
    
    chunks = process_document_folder(
        data_folder,
        min_words_per_page=100,
        target_chunk_size=512,
        min_chunk_size=256,
        overlap_size=150
    )
    
    # Check if we got any chunks at all
    if not chunks:
        print("[WARNING] No document chunks were generated. Check your data folder.")
        # Create a minimal empty vectorstore to avoid errors
        vectorstore = LC_FAISS.from_documents([Document(page_content="Empty index", metadata={})], embeddings)
        vectorstore.save_local(indexes_folder)
        return vectorstore
    
    # Save chunk data into updated_chunks_1.json
    chunks_json = os.path.join(indexes_folder, "updated_chunks_1.json")
    with open(chunks_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("[INFO] Saved chunks to updated_chunks_1.json")

    # Convert chunks to LangChain Document objects
    docs = [Document(page_content=ch["text"], metadata=ch["metadata"]) for ch in chunks]
    # Create LangChain vectorstore from documents
    vectorstore = LC_FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(indexes_folder)
    
    total_chunks = len(chunks)
    unique_docs = len({ch["metadata"]["file_path"] for ch in chunks})
    total_tokens = sum(ch["metadata"].get("token_count", 0) for ch in chunks)
    avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0
    print(f"[INFO] Created {total_chunks} chunks from {unique_docs} documents")
    print(f"[INFO] Average chunk token count: {avg_tokens:.2f}")
    print(f"[INFO] Total number of tokens in all chunks: {total_tokens}")
    return vectorstore

def load_document_metadata_hash(metadata_file_path):
    """Compute an MD5 hash of the document_metadata.json file (i.e. your document fingerprint)."""
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    return None

# Replace the entire load_or_rebuild_vectorstore function with this corrected version:
def load_or_rebuild_vectorstore(data_folder, indexes_folder):
    # Define the file where we'll store our index fingerprint
    fingerprint_file = os.path.join(indexes_folder, "index_fingerprint.json")
    
    # Calculate a stable fingerprint of all documents in the data folder
    current_fingerprint = {}
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.docx', '.pdf', '.txt')):
                path = os.path.join(root, file)
                current_fingerprint[path] = os.path.getmtime(path)
    
    # Sort the fingerprint to ensure consistency
    fingerprint_hash = hashlib.md5(json.dumps(current_fingerprint, sort_keys=True).encode()).hexdigest()
    
    # Check if existing index matches current document state
    previous_fingerprint = None
    if os.path.exists(fingerprint_file):
        try:
            with open(fingerprint_file, 'r') as f:
                previous_fingerprint = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not read fingerprint file: {e}")
    
    # Define paths for the existing index files
    index_path = os.path.join(indexes_folder, "index.faiss")
    metadata_path = os.path.join(indexes_folder, "metadata.json")
    
    print(f"[DEBUG] Current fingerprint: {fingerprint_hash}")
    print(f"[DEBUG] Previous fingerprint: {previous_fingerprint}")
    
    if (os.path.exists(index_path) and 
        os.path.exists(metadata_path) and 
        previous_fingerprint == fingerprint_hash):
        try:
            vectorstore = LC_FAISS.load_local(indexes_folder, embeddings)
            print("[INFO] Loaded existing vectorstore.")
            return vectorstore
        except Exception as e:
            print(f"[INFO] Failed to load existing vectorstore: {e}")
    
    # If we get here, we need to rebuild the index
    print("[INFO] Building/rebuilding index from documents...")
    vectorstore = rebuild_index(data_folder, indexes_folder)
    
    # Save the new fingerprint
    with open(fingerprint_file, 'w') as f:
        json.dump(fingerprint_hash, f)
    
    return vectorstore

# --------------------
# Custom prompt for included sources
# --------------------
def get_qa_prompt_template():
    return """You are a helpful university assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.Write full answer in 2-3 paragraphs.
    
    {context}
    
    Question: {question}
    
    Answer the question clearly and helpfully. DO NOT include any sources, citations, or references at the end of your answer.Write full answer in 2-3 paragraphs.
    """

# --------------------
# Build Conversational Retrieval Chain
# --------------------
# Here we do not pass a ConversationBufferMemory because we are managing history ourselves.
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
if not os.path.exists(INDEXES_FOLDER):
    os.makedirs(INDEXES_FOLDER)
vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER, INDEXES_FOLDER)

# Create a custom prompt that doesn't include sources (we'll add them ourselves)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Create custom QA chain with the new prompt template
qa_prompt = PromptTemplate(
    template=get_qa_prompt_template(),
    input_variables=["context", "question"]
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

# --------------------
# Chat State Management Class
# --------------------
class ChatAssistant:
    """
    This class manages conversation state for the chat interface.
    It maintains a history as a list of dictionaries (with keys "role" and "content"),
    and converts that to the history format expected by the QA chain.
    """
    def __init__(self, qa_chain):
        self.chat_history = []  # List of dictionaries: {"role": "user"/"assistant", "content": ...}
        self.qa = qa_chain

    def _to_chain_history(self):
        """
        Converts the stored chat history into a list of (user, assistant) message pairs,
        which is the format expected by the QA chain.
        """
        pairs = []
        user_text = None
        for msg in self.chat_history:
            if msg["role"] == "user":
                user_text = msg["content"]
            elif msg["role"] == "assistant":
                if user_text is not None:
                    # Strip any "Sources:" section before adding to history
                    answer = msg["content"]
                    if "Sources:" in answer:
                        answer = answer.split("Sources:")[0].strip()
                    pairs.append((user_text, answer))
                    user_text = None
                else:
                    pairs.append(("", msg["content"]))
        return pairs
    
    def _extract_sources(self, source_docs):
        """
        Extract source information from the retrieved documents.
        Returns a list of formatted source citations.
        """
        sources = {}
        for doc in source_docs:
            metadata = doc.metadata
            if "file_name" not in metadata:
                continue
                
            file_name = metadata.get("file_name")
            
            # Track page numbers for PDF files
            if metadata.get("file_type") == "pdf" and "page_number" in metadata:
                if file_name not in sources:
                    sources[file_name] = []
                if metadata["page_number"] not in sources[file_name]:
                    sources[file_name].append(metadata["page_number"])
            else:
                # For non-PDF files, just track the filename
                if file_name not in sources:
                    sources[file_name] = []
        
        # Format the source citations
        formatted_sources = []
        for file_name, pages in sources.items():
            if pages:  # If we have page numbers
                # Sort the page numbers
                pages.sort()
                
                # Group consecutive page numbers
                page_ranges = []
                start = pages[0]
                end = pages[0]
                
                for i in range(1, len(pages)):
                    if pages[i] == end + 1:
                        end = pages[i]
                    else:
                        if start == end:
                            page_ranges.append(str(start))
                        else:
                            page_ranges.append(f"{start}-{end}")
                        start = end = pages[i]
                
                if start == end:
                    page_ranges.append(str(start))
                else:
                    page_ranges.append(f"{start}-{end}")
                
                formatted_sources.append(f"- {file_name} (Page {', '.join(page_ranges)})")
            else:
                formatted_sources.append(f"- {file_name}")
        
        return formatted_sources

    def convchain(self, query):
        if not query:
            return self.chat_history
        # Only process if the new query is different from the last user message
        if not self.chat_history or self.chat_history[-1]["role"] != "user" or self.chat_history[-1]["content"] != query:
            chain_history = self._to_chain_history()
            result = self.qa({"question": query, "chat_history": chain_history})
            answer = result.get("answer", "")
            
            # Extract source information if available and add to the answer
            source_docs = result.get("source_documents", [])
            if source_docs:
                sources = self._extract_sources(source_docs)
                if sources:
                    sources_text = "\n\nSources:\n" + "\n".join(sources)
                    answer += sources_text
            
            # Append the new messages to the history
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
        return self.chat_history

    def clr_history(self):
        self.chat_history = []
        return []

    def get_history_text(self):
        if not self.chat_history:
            return "No conversation history yet."
    
        formatted_history = []
        for msg in self.chat_history:
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            # Remove Sources section from assistant messages in history view
            content = msg["content"]
        if msg["role"] == "assistant" and "Sources:" in content:
                content = content.split("Sources:")[0].strip()
        
        formatted_history.append(f"**{role}**:\n{content}")
    
        return "\n\n".join(formatted_history)
# Instantiate our assistant
assistant = ChatAssistant(qa_chain)

# --------------------
# Gradio Callback Functions
# --------------------
def process_query(query):
    updated_messages = assistant.convchain(query)
    return updated_messages, ""  # Second output resets the input textbox

def clear_history_callback():
    assistant.clr_history()
    return [], "No conversation history yet."
# vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER, INDEXES_FOLDER)
document_manager = initialize_document_manager(DATA_FOLDER)  # <-- ADD THIS LINE

# Instantiate our assistant
assistant = ChatAssistant(qa_chain)
# --------------------
# Gradio Interface
# --------------------
"""
with gr.Blocks(css="" 
    .history-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
        padding: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .history-box p {
        margin-bottom: 10px;
    }
"") as demo: <- добавь сюда один "
    """


with gr.Blocks(css="""
    .history-box {
        border: 1px solid #333;
        border-radius: 10px;
        background-color: #2d2d2d;
        padding: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #f0f0f0;
    }
    .history-box p {
        margin-bottom: 10px;
    }
    .history-box strong {
        color: #ffffff;
    }
""") as demo:
    gr.Markdown("# University Chat Assistant")
    with gr.Tabs():
        with gr.TabItem("Conversation"):
            chatbot = gr.Chatbot(label="Conversation", type="messages", height=500)
            with gr.Row():
                query_input = gr.Textbox(placeholder="Type your message...", show_label=False, container=False)
            #   send_btn = gr.Button("Send")
            
            # Add a clear button for the conversation
           #clear_btn = gr.Button("Clear Chat")
           #clear_btn.click(fn=clear_history_callback, outputs=[chatbot, chatbot])
            
            query_input.submit(fn=process_query, inputs=query_input, outputs=[chatbot, query_input])
           #send_btn.click(fn=process_query, inputs=query_input, outputs=[chatbot, query_input])
            
        with gr.TabItem("Chat History"):
            history_text = gr.Markdown(value="No conversation history yet.", elem_classes=["history-box"])
            with gr.Row():
                refresh_history_btn = gr.Button("Refresh History")
                clear_history_btn = gr.Button("Clear History")
            
            refresh_history_btn.click(fn=lambda: assistant.get_history_text(), outputs=history_text)
            clear_history_btn.click(fn=lambda: (assistant.clr_history(), "No conversation history yet."), outputs=history_text)
        with gr.TabItem("Upload"):
        # Instead of defining everything here, use our function
            document_list = create_upload_tab(document_manager, embeddings, vectorstore, 
                                          lambda: load_or_rebuild_vectorstore(DATA_FOLDER, INDEXES_FOLDER))
    demo.launch(share=True)