import torch
import transformers
import pandas as pd
from data_augmentation import augment_data
from data_preprocess import preprocess_data, return_slots, tokenizer, model
from accuracy import get_accuracy



# Load the ATIS dataset from CSV files
train_data = pd.read_csv('atis_test.csv')
test_data = pd.read_csv('atis_test.csv')

# Convert the data to lists of tuples
train_data = list(zip(train_data['text'], train_data['slots']))
test_data = list(zip(test_data['text'], test_data['slots']))


# Preprocess the train and test data
train_input_ids, train_attention_masks, train_slot_labels = preprocess_data(train_data)
test_input_ids, test_attention_masks, test_slot_labels = preprocess_data(test_data)

# Define the hyperparameters
batch_size = 32
epochs = 10
learning_rate = 2e-5

# Define the data loaders
train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_slot_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks, test_slot_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)



# Move the model to the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model
for epoch in range(epochs):
  # Set the model to training mode
  model.train()
  # Initialize the training loss and accuracy
  train_loss = 0.0
  train_acc = 0.0
  x=0
  # Loop through the batches in the train loader
  for batch in train_loader:
    # Get the input ids, attention masks, and slot labels from the batch
    input_ids = batch[0].to(device)
    attention_masks = batch[1].to(device)
    slot_labels = batch[2].to(device)
    # Forward pass the inputs through the model
    outputs = model(input_ids, attention_mask=attention_masks, labels=slot_labels)
    # Get the loss and logits from the outputs
    loss = outputs.loss
    logits = outputs.logits
    # Backward pass the loss and update the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    # Detach the loss, logits, and slot labels from the device and convert them to numpy arrays
    loss = loss.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    slot_labels = slot_labels.detach().cpu().numpy()
    # Get the predictions from the logits
    preds = torch.argmax(torch.from_numpy(logits), dim=2)
    # Calculate the accuracy for the batch
    acc = get_accuracy(preds, slot_labels)
    # Update the training loss and accuracy
    train_loss += loss
    train_acc += acc
  # Calculate the average training loss and accuracy for the epoch
  train_loss = train_loss / len(train_loader)
  train_acc = train_acc / len(train_loader)
  # Print the training loss and accuracy for the epoch
  print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

# Evaluate the model
# Set the model to evaluation mode
model.eval()
test_loss = 0.0
test_acc = 0.0
# Loop through the batches in the test loader
for batch in test_loader:
  # Get the input ids, attention masks, and slot labels from the batch
  input_ids = batch[0].to(device)
  attention_masks = batch[1].to(device)
  slot_labels = batch[2].to(device)
  # Forward pass the inputs through the model
  outputs = model(input_ids, attention_mask=attention_masks, labels=slot_labels)
  # Get the loss and logits from the outputs
  loss = outputs.loss
  logits = outputs.logits
  # Detach the loss, logits, and slot labels from the device and convert them to numpy arrays
  loss = loss.detach().cpu().numpy()
  logits = logits.detach().cpu().numpy()
  slot_labels = slot_labels.detach().cpu().numpy()
  # Get the predictions from the logits
  preds = torch.argmax(torch.from_numpy(logits), dim=2)
  # Calculate the accuracy for the batch
  acc = get_accuracy(preds, slot_labels)
  # Update the test loss and accuracy
  test_loss += loss
  test_acc += acc
# Calculate the average test loss and accuracy
test_loss = test_loss / len(test_loader)
test_acc = test_acc / len(test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Define a function to test the model on new sentences
def test_model(sentence):
  # Encode the sentence using the BERT tokenizer
  encoding = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
  # Get the input ids and attention mask from the encoding
  input_id = encoding['input_ids'].to(device)
  attention_mask = encoding['attention_mask'].to(device)
  # Forward pass the inputs through the model
  outputs = model(input_id, attention_mask=attention_mask)
  # Get the logits from the outputs
  logits = outputs.logits
  # Detach the logits from the device and convert it to a numpy array
  logits = logits.detach().cpu().numpy()
  # Get the predictions from the logits
  preds = torch.argmax(torch.from_numpy(logits), dim=2)
  # Convert the predictions to a list
  preds = preds.tolist()[0]
  # Remove the special tokens from the predictions
  preds = preds[1:-1]
  output = []
  for pred in preds:
    # If the prediction is not zero, append the corresponding slot label to the output list
    if pred != 0:
      output.append(return_slots[pred-1])
  return output



examples = ["I need to fly from Kansas City to Chicago leaving next Wednesday and returning the following day.",
            "I want to book a flight to Dhaka tomorrow and return to Chittagong this Friday evening.",
            "Show me the cheapest flights from London to Paris"]
for example in examples:
  print(f'Input: {example}')
  print(f'Output: {test_model(example)}')
