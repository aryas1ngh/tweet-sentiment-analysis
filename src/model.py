import torch
from transformers import BertForSequenceClassification
import sys
sys.path.append('../')

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels


    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
            # 'token_type_ids': self.encodings[idx]['token_type_ids']
            }  # Include token_type_ids
        item['labels'] = torch.tensor(self.labels[idx])
    
        return item

    
    def __len__(self):
        return len(self.labels)

def get_model():
    return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
