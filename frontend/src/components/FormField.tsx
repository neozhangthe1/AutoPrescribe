import { FormField as IFormField } from '../lib/types';
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { PlusCircle, Trash2 } from "lucide-react";
import { toast } from "./ui/use-toast";

interface FormFieldProps {
  field: IFormField;
  value: any;
  onChange: (value: any) => void;
}

export const FormField = ({ field, value, onChange }: FormFieldProps) => {
  const renderField = () => {
    console.log('Rendering field:', { type: field.type, id: field.id, value, fields: field.fields });
    
    // Skip fields that aren't in our configuration
    if (!field) {
      return null;
    }

    switch (field.type) {
      case 'dynamic-list': {
        const projectList = Array.isArray(value) ? value : [];
        console.log('Rendering dynamic list:', { projectList, fields: field.fields });
        if (!field.fields) {
          console.error('No fields defined for dynamic-list');
          return null;
        }
        return (
          <div className="space-y-4">
            <div className="flex justify-between items-center mb-4">
              <Label>{field.label}</Label>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  if (projectList.length >= 10) {
                    toast({
                      title: "Error",
                      description: "Maximum of 10 projects allowed",
                      variant: "destructive",
                    });
                    return;
                  }
                  const newList = [...projectList, {}];
                  console.log('Adding new project:', newList);
                  onChange(newList);
                }}
                disabled={projectList.length >= 10}
              >
                <PlusCircle className="h-4 w-4 mr-2" />
                Add Project ({projectList.length}/10)
              </Button>
            </div>
            {projectList.map((project, index) => (
              <Card key={index} className="mb-4">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-lg">Project {index + 1}</CardTitle>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => {
                      const newList = [...projectList];
                      newList.splice(index, 1);
                      console.log('Removing project:', { index, newList });
                      onChange(newList);
                    }}
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Remove
                  </Button>
                </CardHeader>
                <CardContent className="space-y-4 pt-2">
                  {field.fields.map((subField) => (
                    <FormField
                      key={`${index}-${subField.id}`}
                      field={subField}
                      value={project[subField.id] || ''}
                      onChange={(fieldValue) => {
                        const newList = [...projectList];
                        newList[index] = {
                          ...newList[index],
                          [subField.id]: fieldValue
                        };
                        console.log('Updating project field:', { index, field: subField.id, value: fieldValue });
                        onChange(newList);
                      }}
                    />
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        );
      }
      case 'text':
      case 'email':
      case 'tel':
        return (
          <Input
            type={field.type}
            placeholder={field.placeholder}
            value={value || ''}
            onChange={(e) => onChange(e.target.value)}
            required={field.required}
          />
        );
      case 'textarea':
        return (
          <Textarea
            placeholder={field.placeholder}
            value={value || ''}
            onChange={(e) => onChange(e.target.value)}
            required={field.required}
          />
        );
      case 'select':
        return (
          <Select value={value} onValueChange={onChange}>
            <SelectTrigger>
              <SelectValue placeholder="Select..." />
            </SelectTrigger>
            <SelectContent>
              {field.options?.map((option) => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      case 'date':
        return (
          <Input
            type="date"
            value={value || ''}
            onChange={(e) => {
              const date = new Date(e.target.value);
              const formattedDate = date.toISOString().split('T')[0];
              onChange(formattedDate);
            }}
            required={field.required}
          />
        );
      case 'file':
        return (
          <Input
            type="file"
            onChange={(e) => onChange(e.target.files?.[0])}
            required={field.required}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-2">
      <Label>{field.label}{field.required && ' *'}</Label>
      {renderField()}
    </div>
  );
}
