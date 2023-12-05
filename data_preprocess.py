import torch
import transformers
from torch.nn.functional import pad

# Define the slot labels for return date and time entities, including the 'O' label
return_slots = ['O','return_date.day_name', 'return_date.month_name', 'return_date.day_number', 'return_time.period_of_day', 'return_time.time', 'return_time.time_relative'] 

def preprocess_data(data, max_sentence_length=50):
    # Initialize empty lists to store the input ids, attention masks, and slot labels
    input_ids = []
    attention_masks = []
    slot_labels = []
    
    # Loop through the data
    for sentence, slots in data:
        # Encode the sentence using the BERT tokenizer
        encoding = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
        
        # Get the input ids and attention mask from the encoding
        input_id = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Convert the slots to a list of integers
        slot_label = []
        for slot in slots.split():
            try:
                # Try to find the index of the slot label in the return_slots list
                index = return_slots.index(slot)
            except ValueError:
                # If the slot label is not found, assign a default value of zero
                index = 0
            # Append the index to the slot_label list
            slot_label.append(index)
        
        # Convert the list of integers to a tensor
        slot_label = torch.tensor([slot_label])
        
        # Pad or trim the tensors to ensure fixed length
        input_id = pad(input_id, (0, max_sentence_length - input_id.size(1)), value=0)
        attention_mask = pad(attention_mask, (0, max_sentence_length - attention_mask.size(1)), value=0)
        slot_label = pad(slot_label, (0, max_sentence_length - slot_label.size(1)), value=0)
        input_id = input_id[:, :max_sentence_length]
        attention_mask = attention_mask[:, :max_sentence_length]
        slot_label = slot_label[:, :max_sentence_length]
        
        
        # Append the input ids, attention mask, and slot label to the corresponding lists
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        slot_labels.append(slot_label)

    all_input_ids = torch.cat(input_ids, dim=0)
    all_attention_masks = torch.cat(attention_masks, dim=0)
    all_slot_labels = torch.cat(slot_labels, dim=0)

    return all_input_ids, all_attention_masks, all_slot_labels


# Load the BERT tokenizer and model
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(return_slots))