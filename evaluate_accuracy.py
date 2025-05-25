from jiwer import wer
import argparse

def calculate_wer(reference_text, hypothesis_text):
    error = wer(reference_text, hypothesis_text)
    accuracy = (1 - error) * 100
    print(f"Word Error Rate (WER): {error:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    return error, accuracy

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate transcription accuracy using WER.")
    parser.add_argument('--reference', type=str, required=True, help="Path to the reference (ground truth) transcript file")
    parser.add_argument('--hypothesis', type=str, required=True, help="Path to the hypothesis (model output) transcript file")
    args = parser.parse_args()

    reference_text = load_text(args.reference)
    hypothesis_text = load_text(args.hypothesis)

    calculate_wer(reference_text, hypothesis_text)
