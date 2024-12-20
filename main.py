from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from pydantic import BaseModel
from typing import List
from model import train_and_save_model  
from utils import extract_entities_spacy, extract_parameters, normalize_dates_and_times  # Importing utility functions
import os
import json
import pandas as pd
from paraphrase import generate_paraphrases,count_words


# Initialize FastAPI
app = FastAPI()

# Model and labels paths
MODEL_PATH = "intent_model.pth"
LABELS_PATH = "intent_labels.pth"
TOKENIZER_PATH = "intent_tokenizer"

# Global variables for model, tokenizer, and labels
model = None
tokenizer = None
intent_labels = {}


def load_labels():
    global intent_labels
    try:
        intent_labels = torch.load(LABELS_PATH)
        
        logging.info("Labels loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading intent labels: {e}")

def load_tokenizer():
    global tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")

def load_model(intent_labels):
    global model
    try:
        if not intent_labels:
            logging.warning("Intent labels are empty. Cannot initialize model.")
            return
        
        num_labels = len(intent_labels)  # Use the number of labels from intent_labels
        
        # Initialize the BERT model with correct number of labels
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

        # Load state dict with weights_only=True for safety (if applicable)
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")


@app.on_event("startup")
async def startup_event():
    """FastAPI server startup event."""
    load_tokenizer()  # Load tokenizer first.
    
    # Check if LABELS_PATH exists before loading
    if os.path.exists(LABELS_PATH):
        load_labels()  # Load labels
        if intent_labels:  # Only load the model if intent_labels is not empty
            load_model(intent_labels)
        else:
            logging.warning("Loaded intent labels are empty. Skipping model loading.")
    else:
        logging.warning("Labels file not found. Skipping label loading.")
    
    logging.info("FastAPI startup complete.")




@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("uploaded_file.csv", "wb") as f:  # Keep the original file saving
            f.write(contents)

        df = pd.read_csv("uploaded_file.csv")

        augmented_data = []
        for index, row in df.iterrows():
            original_phrase = row['Phrase']
            intent = row['Intent']

            paraphrases = generate_paraphrases(original_phrase)
            augmented_data.append({"Phrase": original_phrase, "Intent": intent})
            
            if count_words(original_phrase) > 2: 
                for paraphrase in paraphrases:
                    augmented_data.append({"Phrase": paraphrase, "Intent": intent})

        json_file_path = "augmented_data.json"  # Or a unique name if needed
        with open(json_file_path, "w") as f:
            json.dump(augmented_data, f)

    # Train the model, passing the JSON file path
        new_intent_labels = train_and_save_model(json_file_path)
        # Now, directly work with the augmented data without saving to a separate file
        

        # Save new intent labels to file
        torch.save(new_intent_labels, LABELS_PATH)

        # Reload model with new intent labels after training
        load_model(new_intent_labels)  # Pass new intent_labels after training.

        return {"message": "Model trained successfully", "labels": new_intent_labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictionRequest(BaseModel):
    texts: List[str]  # Expecting a list of strings

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict intents for a list of input texts."""
    texts = request.texts
    predictions = []

    try:
        logging.info(f"Received texts for prediction: {texts}")  # Log incoming texts
        
        # Tokenize input texts
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
            add_special_tokens=True
        )
        
        logging.info("Tokenization completed successfully.")

        # Perform inference using the loaded model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits and apply softmax to get probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)  # Apply softmax to get probabilities

        # Log logits and probabilities
        logging.info(f"Logits: {logits.tolist()}")  # Convert to list for better readability in logs
        logging.info(f"Probabilities: {probabilities.tolist()}")  # Convert to list for better readability in logs
        
        # Iterate through each phrase's predictions
        for i in range(len(texts)):
            intent_index = torch.argmax(logits[i]).item()
            confidence_level = probabilities[i][intent_index].item()  # Get confidence for predicted class
            
            predicted_intent = [label for label, idx in intent_labels.items() if idx == intent_index]
            
            if predicted_intent:
                predictions.append({
                    "text": texts[i],
                    "predicted_intent": predicted_intent[0],
                    "confidence": confidence_level,
                    "entities": extract_entities_spacy(texts[i]),  # Extract entities using SpaCy
                    "parameters": extract_parameters(texts[i]),      # Extract parameters using regex
                    "normalized_datetime": normalize_dates_and_times(texts[i]).isoformat() if normalize_dates_and_times(texts[i]) else None,
                })
            else:
                logging.warning(f"No predicted intent found for text: {texts[i]}")
                predictions.append({
                    "text": texts[i],
                    "predicted_intent": "Prediction failed.",
                    "confidence": 0.0,
                    "entities": [],
                    "parameters": {},
                    "normalized_datetime": None,
                })
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "predictions": predictions,
        "labels": intent_labels  # Include intent labels in the response
    }
