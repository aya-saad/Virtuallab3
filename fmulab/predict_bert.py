import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification

import numpy as np

import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def predict_bert(sentence, model_path, num_labels=5, max_length=512):
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Define pretrained tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # ----- Predict -----#
    X_test = list([sentence])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=max_length)

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)

    # Load trained model
    # model_path = "static/checkpoint-3000"
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    print('Prediction for: ', sentence, ' is: ', y_pred)
    print('Prediction list: ', sentence, ' is: ', raw_pred)
    return y_pred[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    new_sentence = 'Mocking kung fu pictures when they were a staple of exploitation theater programming was witty'  # 3
    # new_sentence = 'scattered over the course of 80 minutes'  # 1
    # new_sentence = 'introspective and entertaining'  # 3
    num_labels = 5
    max_length = 512
    model_path = "./checkpoint-3000"
    predict_bert(new_sentence, model_path, num_labels, max_length)
