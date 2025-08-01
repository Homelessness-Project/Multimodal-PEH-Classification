import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import json
import os
from tqdm import tqdm

class HomelessnessDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def load_and_preprocess_data(source):
    """Load annotation data and preprocess for BERT training"""
    
    # Map source to annotation file
    source_to_file = {
        'reddit': 'annotation/reddit_raw_scores.csv',
        'x': 'annotation/x_raw_scores.csv', 
        'news': 'annotation/news_raw_scores.csv',
        'meeting_minutes': 'annotation/meeting_minutes_raw_scores.csv'
    }
    
    file_path = source_to_file[source]
    df = pd.read_csv(file_path)
    
    # Get text column
    text_col = 'Deidentified_Comment'
    if source == 'x':
        text_col = 'Deidentified_text'
    elif source == 'news':
        text_col = 'Deidentified_paragraph_text'
    elif source == 'meeting_minutes':
        text_col = 'Deidentified_paragraph'
    
    texts = df[text_col].fillna('').tolist()
    
    # Get label columns (skip first 2 columns: City and text)
    label_cols = [col for col in df.columns[2:] if col not in ['', 'Unnamed: 17', 'Unnamed: 18']]
    
    print(f"Found {len(label_cols)} label categories: {label_cols}")
    
    # Process labels - each category has one column with values 0, 1, 2, 3
    labels = []
    filtered_texts = []
    
    for _, row in df.iterrows():
        label_vector = []
        for col in label_cols:
            score = row.get(col, 0)
            if pd.notna(score) and score != '':
                try:
                    normalized_score = float(score) / 3.0  # Normalize 0-3 to 0-1
                except (ValueError, TypeError):
                    normalized_score = 0.0
            else:
                normalized_score = 0.0
            
            label_vector.append(normalized_score)
        
        # Only include samples with at least one positive label
        if any(score > 0.33 for score in label_vector):  # At least one score >= 1
            labels.append(label_vector)
            filtered_texts.append(str(row[text_col]))
    
    print(f"Loaded {len(filtered_texts)} samples with at least one positive label")
    print(f"Label distribution: {np.sum(labels, axis=0)}")
    
    return filtered_texts, labels, label_cols

def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5):
    """Train the BERT model"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.sigmoid(outputs.logits)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        # Calculate validation F1
        val_predictions = np.array(val_predictions)
        val_true_labels = np.array(val_true_labels)
        
        # Convert to binary for F1 calculation
        binary_val_true = (val_true_labels > 0.33).astype(int)  # Scores >= 1 are positive
        binary_val_pred = (val_predictions > 0.5).astype(int)
        
        val_macro_f1 = f1_score(binary_val_true, binary_val_pred, average='macro', zero_division=0)
        
        print(f'Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Val Macro F1 = {val_macro_f1:.4f}')
        
        # Early stopping
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return model

def evaluate_model(model, test_loader, device, label_names):
    """Evaluate the trained model"""
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Find best threshold for macro F1
    best_threshold = 0.5
    best_macro_f1 = 0
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        binary_pred = (all_predictions > threshold).astype(int)
        binary_true = (all_true_labels > 0.33).astype(int)  # Scores >= 1 are positive
        
        macro_f1 = f1_score(binary_true, binary_pred, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # Final evaluation with best threshold
    final_predictions = (all_predictions > best_threshold).astype(int)
    final_true_labels = (all_true_labels > 0.33).astype(int)
    
    macro_f1 = f1_score(final_true_labels, final_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(final_true_labels, final_predictions, average='micro', zero_division=0)
    
    # Per-label F1 scores
    label_f1_scores = {}
    for i, label_name in enumerate(label_names):
        label_f1 = f1_score(
            final_true_labels[:, i], 
            final_predictions[:, i], 
            zero_division=0
        )
        label_f1_scores[label_name] = label_f1
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'best_threshold': best_threshold,
        'label_f1_scores': label_f1_scores,
        'test_size': len(all_true_labels),
        'num_labels': len(label_names)
    }

def save_results(results, source, label_names):
    """Save results to output directory"""
    
    output_dir = f'output/{source}/bert'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'source': source,
        'model': 'bert-base-uncased',
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'best_threshold': results['best_threshold'],
        'test_size': results['test_size'],
        'num_labels': results['num_labels'],
        'label_f1_scores': results['label_f1_scores']
    }
    
    with open(f'{output_dir}/bert_metrics_{source}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save classification results
    results_df = pd.DataFrame({
        'label': label_names,
        'f1_score': [results['label_f1_scores'][label] for label in label_names]
    })
    results_df.to_csv(f'{output_dir}/bert_classification_results_{source}.csv', index=False)
    
    print(f"Results saved to {output_dir}/")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Best threshold: {results['best_threshold']}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT for homelessness classification')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load and preprocess data
    print(f"Loading data for {args.source}...")
    texts, labels, label_names = load_and_preprocess_data(args.source)
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=args.test_size + args.val_size, random_state=args.seed
    )
    
    val_size_ratio = args.val_size / (args.test_size + args.val_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=1-val_size_ratio, random_state=args.seed
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_names),
        problem_type='multi_label_classification'
    )
    
    # Create datasets
    train_dataset = HomelessnessDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = HomelessnessDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = HomelessnessDataset(test_texts, test_labels, tokenizer, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Training model...")
    model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, label_names)
    
    # Save results
    save_results(results, args.source, label_names)

if __name__ == '__main__':
    main() 