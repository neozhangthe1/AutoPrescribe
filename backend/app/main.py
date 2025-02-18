from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import docx
from docx.shared import Pt
from bs4 import BeautifulSoup
import tempfile
import os
from datetime import datetime
import uuid

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory document storage
documents = {}

def html_to_docx(content: str, title: str) -> str:
    doc = docx.Document()
    doc.add_heading(title, 0)
    
    def process_inline_formatting(text_element, run):
        if text_element.name in ['strong', 'b']:
            run.bold = True
        elif text_element.name in ['em', 'i']:
            run.italic = True
        
        # Process nested formatting
        for child in text_element.children:
            if isinstance(child, str):
                run.text += child.strip()
            elif child.name in ['strong', 'b', 'em', 'i']:
                child_run = run.font
                process_inline_formatting(child, child_run)
    
    soup = BeautifulSoup(content, 'html.parser')
    for element in soup.find_all(['p', 'h1', 'h2', 'ul']):
        if element.name in ['h1', 'h2']:
            level = 1 if element.name == 'h1' else 2
            heading = doc.add_heading(level=level)
            for child in element.children:
                if isinstance(child, str):
                    run = heading.add_run(child.strip())
                else:
                    run = heading.add_run()
                    process_inline_formatting(child, run)
        elif element.name == 'p':
            paragraph = doc.add_paragraph()
            for child in element.children:
                if isinstance(child, str):
                    run = paragraph.add_run(child.strip())
                else:
                    run = paragraph.add_run()
                    process_inline_formatting(child, run)
        elif element.name == 'ul':
            for li in element.find_all('li', recursive=False):
                paragraph = doc.add_paragraph(style='List Bullet')
                for child in li.children:
                    if isinstance(child, str):
                        run = paragraph.add_run(child.strip())
                    else:
                        run = paragraph.add_run()
                        process_inline_formatting(child, run)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
    doc.save(temp_file.name)
    return temp_file.name

@app.post("/documents")
async def create_document(title: str):
    doc_id = str(uuid.uuid4())
    document = {
        "id": doc_id,
        "title": title,
        "content": "",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    documents[doc_id] = document
    return document

@app.get("/documents")
async def list_documents():
    return list(documents.values())

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    return documents[doc_id]

class UpdateDocument(BaseModel):
    content: str

@app.put("/documents/{doc_id}")
async def update_document(doc_id: str, document: UpdateDocument):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    documents[doc_id]["content"] = document.content
    documents[doc_id]["updated_at"] = datetime.utcnow().isoformat()
    return documents[doc_id]

@app.get("/documents/{doc_id}/download")
async def download_document(doc_id: str, background_tasks: BackgroundTasks):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents[doc_id]
    file_path = html_to_docx(document["content"], document["title"])
    
    background_tasks.add_task(os.unlink, file_path)
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{document['title']}.docx"
    )
