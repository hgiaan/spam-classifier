"""
These are some fuctions I use for Naive Bayes spam classifier
- load_data(): Read and label the dataset
    + Read tab-seprated file: <label> <TAB> <message>
    + Returns (message, label) tuples
- preprocess(): clean and tokenise a raw text message
    + Lowercase 
    + Remove punctuations 
    + Remove stop words
    --> Map the same token
"""

import string
STOP_WORDS = set([
    'i', 'me', 'you', 'your', 'he', 'she', 'it', 'we', 'they',
    'my', 'yours', 'his', 'hers',
    'is', 'am', 'are', 'was', 'were', 'will', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'done',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off',
    'get', 'today', 'now', 'just', 'so', 'than', 'too', 'very', 'can', 'this', 'that',
    'u', 's', 'm', 't', 'd', 'll', 're', 've', 'not'
])

def load_data(filename):
    dataset = []
    with open(filename, 'r') as f: 
        for line in f:
            parts = line.strip().split(maxsplit=1) 
            if len(parts) == 2:
                label = parts[0].strip().lower() 
                message = parts[1].strip()
                dataset.append((message, label))
    return dataset

def preprocess(text):
    text = text.lower()
    for char in string.punctuation:
        text = text.replace(char, " ")
    words = text.split()
    return [word for word in words if word not in STOP_WORDS]