import * as React from 'react';
import { useState } from 'react';
import { JobApplicationForm, FormSubmission } from '../lib/types';
import { FormField as FormFieldComponent } from './FormField';
import { api } from '../lib/api';
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { toast } from "./ui/use-toast";

interface ApplicationFormProps {
  form: JobApplicationForm;
}

export function ApplicationForm({ form }: ApplicationFormProps) {
  const [formData, setFormData] = useState<Record<string, any>>({});
  






  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();



    try {
      const submission: FormSubmission = {
        form_id: 'default',
        data: formData
      };
      await api.submitForm(submission);
      toast({
        title: "Success",
        description: "Your application has been submitted successfully!",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to submit the application. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {form.sections.map((section, index) => (
        <Card key={index}>
          <CardHeader>
            <CardTitle>{section.title}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {section.fields.filter(field => field.id !== 'currentRole').map((field) => (
                <FormFieldComponent
                  key={field.id}
                  field={field}
                  value={formData[field.id]}
                  onChange={(value) => {
                    setFormData(prev => ({ ...prev, [field.id]: value }));
                  }}
                />
              ))}
          </CardContent>
        </Card>
      ))}
      <Button type="submit" className="w-full">Submit Application</Button>
    </form>
  );
}
