import argparse
import random
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Lithology Prediction')

    # Data parameters
    parser.add_argument('--dataset', type=str, default='dataset/da.xlsx', help='Dataset to use: da or db')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=189, help='Random seed for reproducibility')

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--conv_channels', type=int, nargs='+', default=[32, 64, 128],
                        help='Channels for conv layers (provide 3 values)')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for conv layers')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=160, help='Number of training epochs')
    parser.add_argument('--label_fractions', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help='Label fractions to test')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs per fraction')

    # Augmentation parameters
    parser.add_argument('--weak_alpha', type=float, default=0.01, help='Weak augmentation noise level')
    parser.add_argument('--strong_alpha', type=float, default=0.1, help='Strong augmentation noise level')
    parser.add_argument('--scale_min', type=float, default=0.95, help='Min scaling for strong augmentation')
    parser.add_argument('--scale_max', type=float, default=1.05, help='Max scaling for strong augmentation')

    # Similarity parameters
    parser.add_argument('--t1', type=float, default=0.05, help='Feature similarity temperature')
    parser.add_argument('--t2', type=float, default=0.05, help='Depth similarity temperature')
    parser.add_argument('--mu', type=float, default=0.9, help='Weight for combining similarities')

    # Pseudo-label parameters
    parser.add_argument('--base_threshold', type=float, default=0.7, help='Base confidence threshold')
    parser.add_argument('--max_threshold', type=float, default=0.9, help='Maximum confidence threshold')
    parser.add_argument('--base_similarity', type=float, default=0.3, help='Base similarity factor')
    parser.add_argument('--min_similarity', type=float, default=0.1, help='Minimum similarity factor')
    parser.add_argument('--consistency_weight', type=float, default=0.1, help='Weight for consistency loss')

    return parser.parse_args()

def setup_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed_all(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device