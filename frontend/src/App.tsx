import { useEffect, useState } from 'react';
import { JobApplicationForm } from './lib/types';
import { ApplicationForm } from './components/ApplicationForm';
import { api } from './lib/api';
import { ToastProvider, ToastViewport } from "./components/ui/toast";
import "./App.css";

function App() {
  const [form, setForm] = useState<JobApplicationForm | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadForm = async () => {
      try {
        const data = await api.getForm('default');
        console.log('Loaded form configuration:', data);
        setForm(data);
      } catch (err) {
        setError('Failed to load the application form');
      }
    };
    loadForm();
  }, []);

  if (error) {
    return <div className="text-red-500">{error}</div>;
  }

  if (!form) {
    return <div>Loading...</div>;
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-8">Job Application Form</h1>
      <ApplicationForm form={form} />
      <ToastProvider>
        <ToastViewport />
      </ToastProvider>
    </div>
  );
}

export default App
