import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import HTTPException

def generate_submission_filename(form_id: str) -> str:
    """Generate a filename for a form submission using form_id with _finish suffix."""
    return f"{form_id}_finish.json"

def save_submission_to_file(data: Dict[str, Any], filepath: str) -> None:
    """Save submission data to a JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save submission: {str(e)}"
        )

def load_submission_from_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load submission data from a JSON file with error handling."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load submission: {str(e)}"
        )
