import * as React from 'react';
import { useState, useEffect } from 'react';
import { JobApplicationForm, FormSubmission, FormField as FormFieldType } from '../lib/types';
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
  
  useEffect(() => {
    // Initialize projects array and clean up any fields not in configuration
    const projectsField = form.sections
      .flatMap(section => section.fields)
      .find(field => field.type === 'dynamic-list');

    const validFields = new Set(
      form.sections.flatMap(section => section.fields.map(field => field.id))
    );

    setFormData(prev => {
      const newData = { ...prev };
      // Remove any fields that aren't in the configuration
      Object.keys(newData).forEach(key => {
        if (!validFields.has(key) || key === 'currentRole') {
          delete newData[key];
        }
      });
      // Initialize projects array if it doesn't exist
      if (projectsField && !newData[projectsField.id]) {
        newData[projectsField.id] = [];
      }
      return newData;
    });
  }, [form]);

  console.log('ApplicationForm render:', { form, formData });
  console.log('Form sections:', form.sections);
  console.log('Experience section:', form.sections.find(section => section.title === 'Experience'));
  console.log('Projects field:', form.sections
    .flatMap(section => section.fields)
    .find(field => field.type === 'dynamic-list'));

  const validateProjectFields = (projects: any[], fields: FormFieldType[]) => {
    return projects.every(project => 
      fields.every(field => !field.required || project[field.id])
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Find the projects field in any section
    const projectsField = form.sections
      .flatMap(section => section.fields)
      .find(field => field.type === 'dynamic-list');

    // Validate projects if they exist
    if (projectsField?.fields && Array.isArray(formData[projectsField.id])) {
      const projects = formData[projectsField.id];
      const hasValidProjects = validateProjectFields(projects, projectsField.fields);

      if (!hasValidProjects) {
        toast({
          title: "Error",
          description: "Please fill in all required fields in project entries.",
          variant: "destructive",
        });
        return;
      }
    }

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
            {section.fields.map((field) => {
              console.log('Rendering field:', field);
              return (
                <FormFieldComponent
                  key={field.id}
                  field={field}
                  value={formData[field.id]}
                  onChange={(value) => {
                    console.log('Field change:', { id: field.id, value });
                    if (field.type === 'dynamic-list') {
                      setFormData(prev => ({ ...prev, [field.id]: value }));
                    } else {
                      setFormData(prev => ({ ...prev, [field.id]: value }));
                    }
                  }}
                />
              );
            })}
          </CardContent>
        </Card>
      ))}
      <Button type="submit" className="w-full">Submit Application</Button>
    </form>
  );
}
