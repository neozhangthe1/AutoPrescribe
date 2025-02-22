import { FormField as IFormField } from '../lib/types';
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";

interface FormFieldProps {
  field: IFormField;
  value: any;
  onChange: (value: any) => void;
}

export const FormField = ({ field, value, onChange }: FormFieldProps) => {
  const renderField = () => {
    if (!field) {
      return null;
    }

    switch (field.type) {
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
