# Named Entity Recognition (NER) using BERT for ATIS Dataset

This is a project to develop a model by finetuning BERT-based NER that can accurately detect the return date and time entities in a given text using the ATIS dataset. The focus is on Named Entity Recognition (NER), specifically targeting return date and time information. The code uses some AI model training practices such as data augmentation, preprocessing, transfer learning, etc. for better accuracy.

> ### Requirements
> - torch: PyTorch library
> - transformers: Hugging Face Transformers library
> - pandas: Data manipulation library
> - spacy: NLP library
> - random: Python standard library for random number generation

## Data
The data used for this project is the ATIS dataset, which contains utterances and their corresponding intents and slot labels for the airline travel information domain. The dataset is available in CSV format from https://huggingface.co/datasets/tuetschek/atis. The slot labels for return date and time entities are defined as follows:

> - return_date.day_name: the name of the day of the week for the return date, e.g. “Monday”, “Tuesday”, etc.
> - return_date.month_name: the name of the month for the return date, e.g. “January”, “February”, etc.
> - return_date.day_number: the number of the day for the return date, e.g. “1st”, “2nd”, etc.
> - return_time.period_of_day: the day for the return time, e.g. “morning”, “afternoon”, etc.
> - return_time.time: the exact time for the return time, e.g. “10:00”, “11:30”, etc.
> - return_time.time_relative: the relative time for the return time, e.g. “in two hours”, “after three days”, etc.

## Code
The code for this project is written in Python and uses the PyTorch framework and the HuggingFace Transformers library. The code consists of the following steps:

> - Import the necessary libraries and modules
> - Load the pre-trained BERT model and tokenizer for NER
> - Load and convert the ATIS dataset to lists of tuples
> - Define a function to augment the data with synonyms
> - Define a function to preprocess the data and convert it to tensors
> - Define a function to calculate the accuracy of the model
> - Define the model parameters and the loss function
> - Fine-tune the model on the train data using the Trainer class
> - Evaluate the model on the test data and report the performance metrics
> - Test the model on some examples and display the results


### To Run the Code
```
$ python main.py
```
