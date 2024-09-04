import os
import argparse
import pickle
import pandas as pd
import string
import PyPDF2  # To read PDF files
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load stopwords and lemmatizer
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Load pre-trained models
with open('model/tfidf_vector.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('model/model_RF.pkl', 'rb') as g:
    model = pickle.load(g)

# Command line arguments
parser = argparse.ArgumentParser(
    description="Categorize resumes using a trained model.")
parser.add_argument("--input_dir", default="input", type=str,
                    help="Path to input directory containing resumes")
parser.add_argument("--output_dir", default="./OUTPUT", type=str,
                    help="Path to output directory for categorized resumes")

args = parser.parse_args()
print(args)


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Removing punctuation and numeric values
    no_punct_tokens = [
        token for token in tokens if token not in string.punctuation and not token.isnumeric()]

    # Removing stop words
    no_stopwords_tokens = [
        token for token in no_punct_tokens if token not in stop_words]

    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(
        token) for token in no_stopwords_tokens]

    # Join tokens back into a string
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def predict(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Transform the text using TF-IDF
    text_vector = tfidf_vectorizer.transform([preprocessed_text])

    # Predict the category using the pre-trained model
    predicted_category = model.predict(text_vector)[0]

    return predicted_category


# Process resumes
resume_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pdf')]
results = []

for resume_file in resume_files:
    resume_path = os.path.join(args.input_dir, resume_file)

    # Extract text from PDF
    resume_content = extract_text_from_pdf(resume_path)

    # Predict the category
    predicted_category = predict(resume_content)

    # Save categorized resume
    output_path = os.path.join(
        args.output_dir, predicted_category, resume_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(open(resume_path, "rb").read())  # Copy the original PDF

    results.append({"resume_file": resume_file,
                    "category": predicted_category})

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(
    args.output_dir, "categorized_resumes.csv"), index=False)
print("Processing completed!")
