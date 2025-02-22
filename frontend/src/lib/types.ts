export interface FormField {
  id: string;
  label: string;
  type: string;  // text, textarea, email, tel, date, select, radio, checkbox, file
  required: boolean;
  placeholder?: string;
  options?: string[];
  validation?: string;
}

export interface FormSection {
  title: string;
  fields: FormField[];
}

export interface JobApplicationForm {
  sections: FormSection[];
}

export interface FormSubmission {
  form_id: string;
  data: Record<string, any>;
}
