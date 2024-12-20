# utils.py
import re
from datetime import datetime
import parsedatetime
import spacy

# Load the SpaCy model (make sure to download the model first)
spacy_nlp = spacy.load("en_core_web_sm")  # Adjust the model as needed

# Initialize the parsedatetime Calendar
cal = parsedatetime.Calendar()

def extract_entities_spacy(text):
    """Extract entities from text using SpaCy."""
    doc = spacy_nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_parameters(text):
    """Extract parameters such as phone numbers and emails using regex patterns."""
    patterns = {
        'phone': r'\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b',  # Matches phone numbers
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Matches email addresses
    }
    
    extracted_info = {}
    
    # Extract using regex patterns
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            extracted_info[key] = matches
    
    return extracted_info

def normalize_dates_and_times(text):
    """Extract and normalize dates and times from text."""
    date_struct, parse_status = cal.parse(text)
    if parse_status:
        return datetime(*date_struct[:6])  # Return a datetime object
    return None

