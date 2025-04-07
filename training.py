import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from data_loader import weak_augmentation, strong_augmentation, compute_feature_and_depth_similarity
from model import *
import random
def adjust_threshold_and_similarity(accuracy, args, base_threshold=None, base_similarity=None,
                                    max_threshold=None, min_similarity=None):
    base_threshold = base_threshold or args.base_threshold
    base_similarity = base_similarity or args.base_similarity
    max_threshold = max_threshold or args.max_threshold
    min_similarity = min_similarity or args.min_similarity

    threshold = base_threshold + (1 - accuracy) * (max_threshold - base_threshold)
    similarity_factor = base_similarity - accuracy * (base_similarity - min_similarity)
    return threshold, similarity_factor

def generate_pseudo_labels_with_similarity(unlabeled_loader, model, X_unlabeled,
                                           depths_unlabeled, accuracy, args, device, t1=None, t2=None):
    model.eval()
    pseudo_data = []
    pseudo_labels = []

    threshold, similarity_factor = adjust_threshold_and_similarity(accuracy, args)
    W_combined = compute_feature_and_depth_similarity(
        X_unlabeled, depths_unlabeled, args, t1, t2)

    with torch.no_grad():
        for batch_idx, inputs in enumerate(unlabeled_loader):
            inputs = inputs[0].to(device)
            weak_aug_inputs = weak_augmentation(inputs, args=args)
            outputs = model(weak_aug_inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, predicted_labels = torch.max(probs, 1)

            for i in range(len(predicted_labels)):
                if max_probs[i] >= threshold:
                    similar_samples = np.where(W_combined[i] > similarity_factor)[0]
                    similar_samples = [j for j in similar_samples if j < len(inputs)]
                    similar_labels = [predicted_labels[j].item() for j in similar_samples]

                    if len(set(similar_labels)) == 1:
                        pseudo_data.append(inputs[i].cpu().numpy())
                        pseudo_labels.append(predicted_labels[i].item())

    return pseudo_data if pseudo_data else None, pseudo_labels if pseudo_labels else None

def train_model_with_consistency(train_loader, model, criterion, optimizer,
                                 X_test_tensor, y_test_tensor, args, device, epochs=None):
    epochs = epochs or args.epochs
    best_test_metrics = {
        'accuracy': -1,
        'precision': -1,
        'recall': -1,
        'f1': -1
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_train_preds = []
        all_train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Consistency regularization
            weak_aug_inputs = weak_augmentation(inputs, args=args)
            weak_outputs = model(weak_aug_inputs)
            strong_aug_inputs = strong_augmentation(inputs, args=args)
            strong_outputs = model(strong_aug_inputs)
            consistency_loss = torch.mean((weak_outputs - strong_outputs) ** 2)
            loss += args.consistency_weight * consistency_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor.to(device))
            _, predicted_test = torch.max(outputs, 1)
            test_accuracy = (predicted_test == y_test_tensor.to(device)).sum().item() / len(y_test_tensor)
            test_precision = precision_score(
                y_test_tensor.cpu().numpy(), predicted_test.cpu().numpy(),
                average='weighted', zero_division=0
            )
            test_recall = recall_score(
                y_test_tensor.cpu().numpy(), predicted_test.cpu().numpy(),
                average='weighted'
            )
            test_f1 = f1_score(
                y_test_tensor.cpu().numpy(), predicted_test.cpu().numpy(),
                average='weighted'
            )

        if (epoch + 1) % 1 == 0:
            train_accuracy = correct_train / total_train
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
                  f"Test  - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
                  f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        if test_accuracy > best_test_metrics['accuracy']:
            best_test_metrics['accuracy'] = test_accuracy
            best_test_metrics['precision'] = test_precision
            best_test_metrics['recall'] = test_recall
            best_test_metrics['f1'] = test_f1

    print("\nBest Test Metrics:")
    print(f"Best Accuracy: {best_test_metrics['accuracy']:.4f}, "
          f"Best Precision: {best_test_metrics['precision']:.4f}, "
          f"Best Recall: {best_test_metrics['recall']:.4f}, "
          f"Best F1: {best_test_metrics['f1']:.4f}")

    return (best_test_metrics['accuracy'], best_test_metrics['precision'],
            best_test_metrics['recall'], best_test_metrics['f1'])

def semi_supervised_train_with_augmented_data(X_train, y_train, X_test, y_test,
                                              X_unlabeled, depths, label_fraction, args, device, seed=None):
    seed = seed or args.random_state
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    labeled_indices = []
    for label in np.unique(y_train):
        class_indices = np.where(y_train == label)[0]
        num_labeled_samples = int(len(class_indices) * label_fraction)
        labeled_class_indices = np.random.choice(class_indices, num_labeled_samples, replace=False)
        labeled_indices.extend(labeled_class_indices)

    unlabeled_indices = list(set(range(len(y_train))) - set(labeled_indices))

    X_train_labeled = X_train[labeled_indices]
    y_train_labeled = y_train[labeled_indices]
    X_train_unlabeled = X_train[unlabeled_indices]

    X_train_labeled_tensor = torch.tensor(X_train_labeled, dtype=torch.float32).to(device)
    y_train_labeled_tensor = torch.tensor(y_train_labeled, dtype=torch.long).to(device)
    X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    labeled_dataset = TensorDataset(X_train_labeled_tensor, y_train_labeled_tensor)
    unlabeled_dataset = TensorDataset(X_train_unlabeled_tensor)

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = LithologyModel(args, input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"\nTraining with {int(label_fraction * 100)}% labeled data (Initial Supervised Training)...")
    test_metrics = train_model_with_consistency(
        labeled_loader, model, criterion, optimizer, X_test_tensor, y_test_tensor, args, device)
    print(f"\nGenerating pseudo-labels with weak augmentation...")
    pseudo_data, pseudo_labels = generate_pseudo_labels_with_similarity(
        unlabeled_loader, model, X_train_unlabeled, depths, test_metrics[0], args, device)

    if not pseudo_data:
        print("No pseudo-labeled samples met the threshold. Retraining with labeled data only.")
        pseudo_data, pseudo_labels = X_train_labeled, y_train_labeled

    pseudo_data_tensor = torch.tensor(np.array(pseudo_data), dtype=torch.float32).to(device)
    pseudo_labels_tensor = torch.tensor(np.array(pseudo_labels), dtype=torch.long).to(device)

    pseudo_dataset = TensorDataset(pseudo_data_tensor, pseudo_labels_tensor)
    full_train_dataset = torch.utils.data.ConcatDataset([labeled_dataset, pseudo_dataset])
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    print("\nRetraining with pseudo-labeled data...")
    test_metrics = train_model_with_consistency(
        full_train_loader, model, criterion, optimizer, X_test_tensor, y_test_tensor, args, device)

    return test_metrics