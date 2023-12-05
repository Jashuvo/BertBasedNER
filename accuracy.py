import torch

# Define a function to calculate the accuracy
def get_accuracy(preds, labels):
  # Convert the predictions and labels to numpy arrays
  preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
  labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
  # Flatten the predictions and labels
  preds = preds.flatten()
  labels = labels.flatten()
  # Count the number of correct predictions
  correct = (preds == labels).sum().item()
  # Count the total number of predictions
  total = len(labels)
  # Calculate and return the accuracy
  accuracy = correct / total
  return accuracy