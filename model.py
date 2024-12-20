import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AdamW
import pandas as pd
from sklearn.metrics import classification_report
import logging
import json 

MODEL_PATH = "intent_model.pth"  # Save the full model
TOKENIZER_PATH = "intent_tokenizer"  # Save tokenizer
LABELS_PATH = "intent_labels.pth"  # Save intent labels

def train_and_save_model(file_path):

    try:
        if file_path.endswith(".json"):  # Load from JSON if specified
            with open(file_path, 'r') as f:
                data = json.load(f)
                data = pd.DataFrame(data)
        elif file_path.endswith(".csv"):  # Otherwise, assume CSV
            data = pd.read_csv(file_path)
        else:
            raise ValueError("Invalid file path. Must be a CSV or JSON file.")

        print("Columns in the dataset:", data.columns.tolist())
        if "Phrase" not in data.columns or "Intent" not in data.columns:
            logging.error("Missing required columns in input data.")
            return {}
        
        data.dropna(subset=["Phrase", "Intent"], inplace=True)

        if data.empty:
            logging.error("The dataset is empty after dropping NA values.")
            return {}
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            data["Phrase"], data["Intent"], test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Tokenizer setup
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Encode phrases
        def encode_data(texts):
            return tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )

        # Tokenize
        train_encodings = encode_data(X_train)
        val_encodings = encode_data(X_val)

        # Map labels
        intent_labels = {label: idx for idx, label in enumerate(data["Intent"].unique())}
        y_train_encoded = torch.tensor([intent_labels[intent] for intent in y_train])
        y_val_encoded = torch.tensor([intent_labels[intent] for intent in y_val])

        # Dataset construction
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            y_train_encoded,
        )
        
        val_dataset = TensorDataset(
            val_encodings["input_ids"],
            val_encodings["attention_mask"],
            y_val_encoded,
        )

        # Model setup
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=len(intent_labels)
        )

        optimizer = AdamW(model.parameters(), lr=2e-5)

        # Early stopping parameters
        patience = 3  # Number of epochs to wait for improvement
        best_val_loss = float('inf')
        patience_counter = 0

        # Lists for logging
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Training loop
        for epoch in range(15):  # Reduced epochs for testing
            model.train()
            total_loss = 0
            correct_predictions = 0
            
            for batch in DataLoader(train_dataset, batch_size=8, shuffle=True):
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()

            avg_train_loss = total_loss / len(train_dataset)
            avg_train_accuracy = correct_predictions / len(train_dataset)

            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_accuracy)

            # Validation phase
            model.eval()
            total_val_loss = 0
            val_correct_predictions = 0

            with torch.no_grad():
                for batch in DataLoader(val_dataset, batch_size=8):
                    input_ids, attention_mask, labels = batch

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    val_loss = outputs.loss
                    total_val_loss += val_loss.item()

                    preds = torch.argmax(outputs.logits, dim=-1)
                    val_correct_predictions += (preds == labels).sum().item()

            avg_val_loss = total_val_loss / len(val_dataset)
            avg_val_accuracy = val_correct_predictions / len(val_dataset)

            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)

            print(f"Epoch {epoch + 1}/{15} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0  # Reset counter if we have an improvement
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break  # Exit the training loop if no improvement is seen

        # Save the entire model after training
        model.eval()
        torch.save(model, MODEL_PATH)  # Save the entire model state dict
        tokenizer.save_pretrained(TOKENIZER_PATH)  # Save tokenizer
        torch.save(intent_labels, LABELS_PATH)  # Save intent label mapping
        
        print("Model, tokenizer, and intent labels saved successfully.")

        # Generate classification report on validation set
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in DataLoader(val_dataset):
                input_ids, attention_mask, labels = batch
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_labels, all_preds))

        return intent_labels  # Return the intent labels

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return {}
