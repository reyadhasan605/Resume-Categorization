import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle

class ResumeDataset(Dataset):
    def __init__(self, resumes, labels, tokenizer, max_length):
        self.resumes = resumes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.resumes)

    def __getitem__(self, idx):
        resume = str(self.resumes[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            resume,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv('./dataset/Resume.csv')

label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category'])

# Prepare datasets and dataloaders
max_length = 128
train_dataset = ResumeDataset(train_df['Resume_str'].values, train_df['Category'].values, tokenizer, max_length)
val_dataset = ResumeDataset(val_df['Resume_str'].values, val_df['Category'].values, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Category'].unique()))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = AdamW(model.parameters(), lr=2e-5)

def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    total_loss = 0

    for batch in tqdm(data_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model = model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(data_loader), accuracy, predictions, true_labels

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_accuracy, _, _ = eval_model(model, val_loader, device)
    print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

_, _, predictions, true_labels = eval_model(model, val_loader, device)
print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))


model_path = './model/Bert_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

label_encoder_path = './model/Bert_model/label_encoder.pkl'

# Save the LabelEncoder
with open(label_encoder_path, 'wb') as file:
    pickle.dump(label_encoder, file)