import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from .models import JobApplicationForm, FormSubmission, FormSection, FormField
from .config import settings
from .utils import generate_submission_filename, save_submission_to_file, load_submission_from_file

app = FastAPI(title=settings.app_name)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File storage paths
SUBMISSIONS_DIR = "data/submissions"
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Form storage
import os
import json
from pathlib import Path

FORMS_DIR = Path("forms")

forms: Dict[str, JobApplicationForm] = {}

def load_form_config(form_id: str) -> JobApplicationForm:
    """Load form configuration from JSON file."""
    try:
        form_path = FORMS_DIR / f"{form_id}.json"
        if not form_path.is_file():
            raise FileNotFoundError(f"Form configuration '{form_id}' not found")
        
        with open(form_path) as f:
            form_data = json.load(f)
            return JobApplicationForm(**form_data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Form configuration '{form_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading form configuration: {str(e)}")

# Load default form configuration
forms["default"] = load_form_config("default")

@app.get("/api/forms/{form_id}", response_model=JobApplicationForm)
async def get_form(form_id: str):
    try:
        if form_id not in forms:
            # Try to load the form if it hasn't been loaded yet
            forms[form_id] = load_form_config(form_id)
        return forms[form_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading form: {str(e)}")

@app.post("/api/submissions/")
async def submit_form(submission: FormSubmission):
    try:
        filename = generate_submission_filename(submission.form_id)
        filepath = os.path.join(SUBMISSIONS_DIR, filename)
        save_submission_to_file(submission.dict(), filepath)
        return {
            "status": "success",
            "message": "Form submitted successfully",
            "file": filename
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/api/submissions/{form_id}")
async def get_submission(form_id: str):
    filename = generate_submission_filename(form_id)
    filepath = os.path.join(SUBMISSIONS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Submission not found")
    return load_submission_from_file(filepath)

@app.get("/api/submissions/")
async def get_submissions():
    submissions = {}
    try:
        for filename in os.listdir(SUBMISSIONS_DIR):
            if filename.endswith("_finish.json"):
                form_id = filename.replace("_finish.json", "")
                filepath = os.path.join(SUBMISSIONS_DIR, filename)
                submission = load_submission_from_file(filepath)
                if submission:
                    submissions[form_id] = submission
        return submissions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list submissions: {str(e)}"
        )
