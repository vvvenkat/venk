import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Paths for your model, labels, and tokenizer
MODEL_PATH = "intent_model.pth"
LABELS_PATH = "intent_labels.pth"
TOKENIZER_PATH = "intent_tokenizer"

# Load the labels
def load_labels():
    try:
        labels = torch.load(LABELS_PATH)
        print("Labels loaded successfully.")
        return labels
    except Exception as e:
        print(f"Error loading intent labels: {e}")
        return {}

# Load the tokenizer
def load_tokenizer():
    try:
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        print("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

# Load the model
def load_model(intent_labels):
    try:
        # Initialize the model with correct number of labels
        num_labels = len(intent_labels)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        
        # Load state dict with weights_only=True for safety
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to predict intents and confidence levels for a list of phrases
def predict_intents(phrases, model, tokenizer, intent_labels):
    predictions = []
    
    try:
        # Tokenize input texts
        inputs = tokenizer(
            phrases,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
            add_special_tokens=True  # Add [CLS] and [SEP]
        )

        # Perform inference using the loaded model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits and apply softmax to get probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)  # Apply softmax to get probabilities
        
        # Iterate through each phrase's predictions
        for i in range(len(phrases)):
            intent_index = torch.argmax(logits[i]).item()
            confidence_level = probabilities[i][intent_index].item()  # Get confidence for predicted class
            
            # Map index to label using intent_labels dictionary
            predicted_intent = [label for label, idx in intent_labels.items() if idx == intent_index]
            
            if predicted_intent:
                predictions.append((predicted_intent[0], confidence_level))
            else:
                predictions.append(("Prediction failed.", 0.0))
    
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    return predictions

if __name__ == "__main__":
    # Load labels first to know how many labels to set in the model
    intent_labels = load_labels()  
    
    # Load the model after loading labels
    model = load_model(intent_labels)  
    tokenizer = load_tokenizer()

    # List of phrases for prediction
    phrases_to_predict = [
    "Hi"
]


    # Make predictions for all phrases in the list
    if model and tokenizer and intent_labels:
        predictions = predict_intents(phrases_to_predict, model, tokenizer, intent_labels)
        
        for phrase, (predicted_label, confidence_level) in zip(phrases_to_predict, predictions):
            print(f"Predicted intent for '{phrase}': {predicted_label}")
            print(f"Confidence level: {confidence_level:.2f}")  # Format confidence level to two decimal places
