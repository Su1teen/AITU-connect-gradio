import os
import gradio as gr
import tempfile
from document_manager import DocumentManager

def initialize_document_manager(data_folder):
    """Initialize and return a document manager instance for the data folder."""
    manager = DocumentManager(data_folder)
    # Only initialize from existing files if the metadata file doesn't exist
    if not os.path.exists(os.path.join(data_folder, "document_metadata.json")):
        manager.initialize_from_existing_files()
    return manager

def filter_documents(document_manager, search_term):
    """Filter documents based on search term."""
    docs = document_manager.get_active_documents()
    if search_term:
        search_term = search_term.lower()
        docs = [doc for doc in docs if (
            search_term in doc["title"].lower() or 
            search_term in doc["original_filename"].lower() or
            any(search_term in tag.lower() for tag in doc["tags"])
        )]
    return [[
        doc["id"], 
        doc["original_filename"], 
        doc["title"], 
        doc["upload_date"].split("T")[0], 
        doc["version"]
    ] for doc in docs]

def find_similar_docs(file_obj, document_manager, embeddings, vectorstore):
    """Find documents similar to the uploaded file."""
    if not file_obj:
        return "## Similar Documents\nUpload a file to see similar existing documents."
    
    similar_docs = document_manager.find_similar_documents(file_obj.name, embeddings, vectorstore)
    
    if not similar_docs:
        return "## Similar Documents\nNo similar documents found."
    
    result = "## Similar Documents\nThese existing documents appear similar to your upload:\n\n"
    for doc, score in similar_docs:
        result += f"- **{doc['title']}** ({doc['original_filename']}) - {score:.2f} similarity score\n"
    
    return result

def get_document_preview(file_obj):
    """Get a preview of the document content."""
    if not file_obj:
        return "Upload a file to see preview."
    try:
        from document_manager import DocumentManager
        temp_manager = DocumentManager("temp")
        preview = temp_manager.extract_preview(file_obj.name)
        return preview or "No preview available."
    except Exception as e:
        return f"Error generating preview: {str(e)}"

def get_comparison_preview(file_obj, doc_id, document_manager):
    """Get a comparison between the uploaded file and the selected document."""
    if not file_obj or not doc_id:
        return "Select both a file to upload and an existing document to compare."
    try:
        new_preview = document_manager.extract_preview(file_obj.name)
        existing_doc = document_manager.get_document_by_id(doc_id)
        if not existing_doc:
            return "Selected document not found."
        existing_preview = existing_doc.get("preview", "")
        result = f"## Document Comparison\n\n"
        result += f"### New Document\n{new_preview[:300]}...\n\n"
        result += f"### Existing Document: {existing_doc['title']}\n{existing_preview[:300]}...\n\n"
        return result
    except Exception as e:
        return f"Error generating comparison: {str(e)}"

def process_upload(file_obj, title, description, tags_str, is_update, update_doc_id, document_manager, rebuild_fn, embeddings, vectorstore):
    """Process the document upload, handling both new and update cases."""
    if not file_obj:
        return "Please select a file to upload."
    try:
        tags = [tag.strip() for tag in tags_str.split(",")] if tags_str else []
        if is_update:
            similar_docs = document_manager.find_similar_documents(file_obj.name, embeddings, vectorstore)
            if similar_docs:
                # Assume the first result is the most similar; you could sort by score if desired.
                most_similar_doc, score = similar_docs[0]
                document_manager.delete_document_by_id(most_similar_doc["id"])
                print(f"Deleted document: {most_similar_doc['title']} as an update.")
            else:
                return "No similar document found to update."
        
        doc = document_manager.add_document(
            file_obj.name,
            title,
            description,
            tags,
            is_update,
            update_doc_id if is_update and update_doc_id else None
        )
        rebuild_fn()
        return f"✅ Document uploaded successfully: {doc['title']} (Version {doc['version']})"
    except Exception as e:
        return f"❌ Error uploading document: {str(e)}"

def create_upload_tab(document_manager, embeddings, vectorstore, rebuild_fn):
    """Create and return the upload tab component."""
    with gr.TabItem("Upload"):
        gr.Markdown("# Upload Documents")
        with gr.Row():
            upload_file = gr.File(label="Upload Document")
        with gr.Column():
            doc_preview = gr.Markdown("Upload a file to see preview.")
        with gr.Row():
            doc_title = gr.Textbox(label="Document Title (optional)")
            doc_description = gr.Textbox(label="Description (optional)")
            doc_tags = gr.Textbox(label="Tags (comma separated, optional)")
        with gr.Row():
            is_update = gr.Checkbox(label="This is an update to an existing document")
        # Wrap the update section in a container that is updatable
        with gr.Column(visible=False) as update_container:
            with gr.Accordion("Select document to update", open=True):
                search_box = gr.Textbox(label="Search documents")
                doc_list = gr.Dataframe(
                    headers=["ID", "Filename", "Title", "Upload Date", "Version"],
                    datatype=["str", "str", "str", "str", "number"],
                    label="Existing Documents"
                )
                selected_doc_id = gr.Textbox(visible=False)
                comparison_preview = gr.Markdown("Select a document to see comparison.")
        with gr.Row():
            similar_docs_md = gr.Markdown("## Similar Documents\nUpload a file to see similar existing documents.")
        with gr.Row():
            upload_button = gr.Button("Upload Document")
            upload_status = gr.Markdown("Ready to upload.")
        
        # Initialize document list
        doc_list.value = filter_documents(document_manager, "")
        
        # Update container visibility based on is_update checkbox
        is_update.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[is_update],
            outputs=[update_container]
        )
        
        # Searching documents
        search_box.change(
            fn=lambda x: filter_documents(document_manager, x),
            inputs=[search_box],
            outputs=[doc_list]
        )
        
        # Selecting a document
        doc_list.select(
            fn=lambda x, y: (x[0] if x else "", 
                             get_comparison_preview(y, x[0] if x else "", document_manager)),
            inputs=[doc_list, upload_file],
            outputs=[selected_doc_id, comparison_preview]
        )
        
        # Show document preview on upload
        upload_file.change(
            fn=get_document_preview,
            inputs=[upload_file],
            outputs=[doc_preview]
        )
        
        # Run similarity search on upload
        upload_file.change(
            fn=lambda x: find_similar_docs(x, document_manager, embeddings, vectorstore),
            inputs=[upload_file],
            outputs=[similar_docs_md]
        )
        
        # Process upload
        upload_button.click(
            fn=lambda file, title, desc, tags, is_upd, doc_id: process_upload(
                file, title, desc, tags, is_upd, doc_id, document_manager, rebuild_fn, embeddings, vectorstore
            ),
            inputs=[upload_file, doc_title, doc_description, doc_tags, is_update, selected_doc_id],
            outputs=[upload_status]
        )
    
    return doc_list
