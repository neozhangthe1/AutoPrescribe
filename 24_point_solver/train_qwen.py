import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
from tqdm import tqdm

class ArithmeticDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        # Format data for training
        self.formatted_data = []
        for problem in self.data:
            # Format input
            numbers = problem['numbers']
            input_text = f"Calculate 24: Given numbers {numbers}, find operations to make 24.\n"
            
            # Format solution steps
            solution_steps = "\n".join(problem['steps'])
            
            # Combine input and solution
            full_text = f"{input_text}\nSolution:\n{solution_steps}\n"
            self.formatted_data.append(full_text)
    
    def __len__(self):
        return len(self.formatted_data)
    
    def __getitem__(self, idx):
        text = self.formatted_data[idx]
        
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare model inputs
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        # Mask padding tokens
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def compute_metrics(eval_pred):
    """Custom metrics for 24-point problem solving"""
    predictions, labels = eval_pred
    # Convert logits to predictions
    predictions = np.argmax(predictions, axis=-1)
    
    # Calculate accuracy only on non-masked tokens
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    return {
        'accuracy': accuracy
    }

def train_qwen(
    model_name="Qwen/Qwen-0.5B",
    data_path="sample_problems.json",
    output_dir="qwen_24point_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_grad_norm=1.0,
    warmup_ratio=0.1
):
    """Train Qwen model on 24-point arithmetic problems"""
    
    # Load tokenizer and model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use fp16 for efficiency
        device_map="auto"  # Automatically handle device placement
    )
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = ArithmeticDataset(data_path, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        fp16=True,  # Enable mixed precision training
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer, model, tokenizer

def evaluate_model(model, tokenizer, test_problems, max_length=2048):
    """Evaluate model on test problems"""
    model.eval()
    results = []
    
    for problem in tqdm(test_problems, desc="Evaluating"):
        # Format input
        numbers = problem['numbers']
        input_text = f"Calculate 24: Given numbers {numbers}, find operations to make 24.\n"
        
        # Generate solution
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode generated solution
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Store results
        results.append({
            'input': input_text,
            'generated': generated_text,
            'reference': problem['steps']
        })
    
    return results

if __name__ == "__main__":
    # Train model
    trainer, model, tokenizer = train_qwen()
    
    # Load test problems
    with open("sample_problems.json", 'r') as f:
        test_problems = json.load(f)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    results = evaluate_model(model, tokenizer, test_problems)
    
    # Save evaluation results
    with open("evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training and evaluation completed!")
