import spacy
import random


# Define a function to augment the data with synonyms
def augment_data(data):
  # Load the spaCy model for English
  nlp = spacy.load("en_core_web_sm")
  # Define some synonyms for common words
  synonyms = {'fly': ['travel', 'go', 'take a flight'], 'book': ['reserve', 'make a reservation', 'get a ticket'], 'return': ['come back', 'go back', 'fly back']}
  # Initialize an empty list to store the augmented data
  augmented_data = []
  # Loop through the data
  for sentence, slots in data:
    # Convert the sentence to a spaCy document
    doc = nlp(sentence)
    # Initialize an empty list to store the new sentence tokens
    new_sentence = []
    # Loop through the tokens in the document
    for token in doc:
      # If the token text is in the synonyms dictionary, randomly choose a synonym
      if token.text in synonyms:
        new_word = random.choice(synonyms[token.text])
      # Otherwise, keep the original token text
      else:
        new_word = token.text
      # Append the new word to the new sentence list
      new_sentence.append(new_word)
    # Join the new sentence list into a string
    new_sentence = ' '.join(new_sentence)
    # Append the new sentence and the original slots to the augmented data list
    augmented_data.append((new_sentence, slots))
  # Return the augmented data list
  return augmented_data