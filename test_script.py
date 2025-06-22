import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(csv_path):
    df = pd.read_csv(csv_path)

    # Assuming your results.csv has a 'label' column (true) and 'similarity' column
    true = df['label'].values
    pred = [1 if s >= 0.90 else 0 for s in df['similarity'].values]

    print("✅ Evaluation Results:")
    print(f"Accuracy:  {accuracy_score(true, pred):.2f}")
    print(f"Precision: {precision_score(true, pred):.2f}")
    print(f"Recall:    {recall_score(true, pred):.2f}")
    print(f"F1 Score:  {f1_score(true, pred):.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_script.py path/to/task_b_results.csv")
    else:
        evaluate(sys.argv[1])
