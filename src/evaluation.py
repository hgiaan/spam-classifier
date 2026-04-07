from utils import load_data
from model import naive_bayes, predict

def evaluate(dataset, test_ratio=0.2):
    split = int(len(dataset) * (1 - test_ratio))
    train_set = dataset[:split]
    test_set  = dataset[split:]

    model = naive_bayes(train_set)

    correct = 0
    for message, true_label in test_set:
        prediction = predict(message, model)
        pred_label = "spam" if "SPAM" in prediction else "ham"
        if pred_label == true_label:
            correct += 1

    accuracy = correct / len(test_set) * 100
    print(f"\n--- EVALUATION ---")
    print(f"Test set size : {len(test_set)} messages")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.1f}%")

if __name__ == "__main__":
    try:
        # Đường dẫn tới file data tương tự như trong main.py
        messages_data = load_data('../data/spamhamdata.csv')
        evaluate(messages_data)
    except FileNotFoundError:
        print("Error: Could not find 'spamhamdata.csv'.")