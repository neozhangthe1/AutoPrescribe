from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, validator
from datetime import datetime

class FormField(BaseModel):
    id: str
    label: str
    type: str  # text, textarea, email, tel, date, select, radio, checkbox, file, dynamic-list
    required: bool = False
    placeholder: Optional[str] = None
    options: Optional[List[str]] = None  # For select, radio, checkbox
    validation: Optional[str] = None  # regex pattern for validation
    fields: Optional[List['FormField']] = None  # For dynamic-list type, contains fields for each item

class FormSection(BaseModel):
    title: str
    fields: List[FormField]

class JobApplicationForm(BaseModel):
    sections: List[FormSection]

class FormSubmission(BaseModel):
    form_id: str
    data: Dict[str, Any]
    
    @validator('data')
    def validate_required_fields(cls, v, values):
        from .main import forms  # Import here to avoid circular import
        if 'form_id' in values:
            form = forms.get(values['form_id'])
            if form:
                for section in form.sections:
                    for field in section.fields:
                        if field.required and field.id not in v:
                            raise ValueError(f"Required field '{field.id}' is missing")
                        if field.id in v and field.type == 'email':
                            if '@' not in str(v[field.id]):
                                raise ValueError(f"Invalid email format for field '{field.id}'")
        return v
