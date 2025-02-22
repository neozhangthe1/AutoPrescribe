import axios from 'axios';
import { JobApplicationForm, FormSubmission } from './types';

declare global {
  interface ImportMetaEnv {
    VITE_API_URL: string;
  }
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SubmissionResponse {
  status: string;
  message: string;
  file: string;
}

export const api = {
  getForm: async (formId: string): Promise<JobApplicationForm> => {
    const response = await axios.get(`${API_URL}/api/forms/${formId}`);
    return response.data;
  },

  submitForm: async (submission: FormSubmission): Promise<SubmissionResponse> => {
    const response = await axios.post(`${API_URL}/api/submissions/`, submission);
    return response.data;
  },

  getSubmission: async (filename: string): Promise<FormSubmission> => {
    const response = await axios.get(`${API_URL}/api/submissions/${filename}`);
    return response.data;
  },

  getAllSubmissions: async (): Promise<FormSubmission[]> => {
    const response = await axios.get(`${API_URL}/api/submissions/`);
    return response.data;
  }
};
