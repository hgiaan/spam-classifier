from utils import load_data
from model import naive_bayes, predict
 
try:
    messages_data = load_data('../data/spamhamdata.csv')
    print(f"Loaded {len(messages_data)} messages from spamhamdata.csv\n")
 
    trained_model = naive_bayes(messages_data)
 
    test_messages = [
        "Hey, are you free this afternoon to study at the library?",
        "Get a 50% discount voucher now, click the link!",
        "ofjoaiwjer"
    ]
 
    print("Results: ")
    for msg in test_messages:
        print(f"Message: '{msg}'\n=> Prediction: {predict(msg, trained_model)}\n")
 
except FileNotFoundError:
    print("Error: Could not find 'spamhamdata.csv'.")