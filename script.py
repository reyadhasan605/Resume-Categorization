import os
import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
import shutil
import pandas as pd
import pickle



parser = argparse.ArgumentParser(
    description="Categorize resumes using a trained model.")
parser.add_argument("--input_dir", default="./input_cv", type=str,
                    help="Path to input directory containing resumes")
parser.add_argument("--output_dir", default = "./OUTPUT",type=str,
                    help="Path to output directory for categorized resumes")

args = parser.parse_args()
print(args)

# Function to load the model
def load_model(model_path, num_labels):
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model

# Function to classify a single resume
def classify_resume(model, tokenizer, resume_text, max_length=128):
    encoding = tokenizer.encode_plus(
        resume_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors='pt',
        padding='max_length'
    )

    input_ids = encoding['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
    attention_mask = encoding['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return prediction

# Function to process all resumes in the INPUT directory
def process_resumes(input_dir, output_dir, model, tokenizer, label_encoder):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt") or filename.endswith(".pdf"):  # Assuming resumes are in .txt or .pdf format
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                resume_text = file.read()

            # Classify the resume
            predicted_label_idx = classify_resume(model, tokenizer, resume_text)
            predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]

            # Create the category directory if it doesn't exist
            category_dir = os.path.join(output_dir, predicted_label)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

            # Move the file to the corresponding category folder
            shutil.move(filepath, os.path.join(category_dir, filename))
            print(f"Moved {filename} to {predicted_label} category.")

            results.append({"resume_file": filename,
                            "category": predicted_label})

            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(
                "./OUTPUT", "categorized_resumes.csv"), index=False)
            print("Processing completed!")



with open('./model/Bert_model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


model_path = './model/Bert_model'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = load_model(model_path, num_labels=len(label_encoder.classes_))


# Set directories
input_dir = args.input_dir
output_dir = args.output_dir

process_resumes(input_dir, output_dir, model, tokenizer, label_encoder)
