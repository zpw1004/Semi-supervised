import numpy as np
import torch
import random
from config import parse_args, setup_device
from data_loader import load_and_preprocess_data
from training import semi_supervised_train_with_augmented_data
from utils import setup_seed

def main():
    args = parse_args()
    device = setup_device(args)
    setup_seed(args.random_state)
    X_train, X_test, y_train, y_test, depths = load_and_preprocess_data(args)
    results = {}
    for fraction in args.label_fractions:
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        for run in range(args.num_runs):
            seed = args.random_state + run
            print(f"\nTraining with {int(fraction * 100)}% labeled data, Run {run + 1}/{args.num_runs}...")
            accuracy, precision, recall, f1 = semi_supervised_train_with_augmented_data(
                X_train, y_train, X_test, y_test, X_train, depths, fraction, args, device, seed=seed)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)
        results[fraction] = (avg_accuracy, avg_precision, avg_recall, avg_f1)
        print(f"\nAverage results for {int(fraction * 100)}% labeled data: "
              f"Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, "
              f"Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    print("\nFinal Results:")
    for fraction, (accuracy, precision, recall, f1) in results.items():
        print(f"Label Fraction: {int(fraction * 100)}%, "
              f"Average Accuracy: {accuracy:.4f}, "
              f"Average Precision: {precision:.4f}, "
              f"Average Recall: {recall:.4f}, "
              f"Average F1: {f1:.4f}")

if __name__ == "__main__":
    main()