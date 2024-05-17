import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import pickle
import sys,os
sys.path.append('../')
class DataProcessor:
    def __init__(self, file_path, model_name='bert-base-uncased', test_size=0.2, n_samples_per_class=None, max_token_length=128, random_state=42, ):
        self.file_path = file_path
        self.model_name = model_name
        self.test_size_val = test_size
        self.n_samples_per_class = n_samples_per_class
        self.max_token_length = max_token_length
        self.random_state = random_state
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.data = self.load_data()

    def test_size(self):
        return self.test_size_val
    
    def load_data(self):
        data = pd.read_csv(self.file_path, encoding='latin-1', header=None, usecols=[0, 5])
        data.columns = ['sentiment', 'text']
        data['sentiment'] = data['sentiment'].map({0: 0, 4: 1})
        return data

    def preprocess_data(self):
        if self.n_samples_per_class:
            self.data = self.sample_data(self.data, self.n_samples_per_class)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            self.data['text'].tolist(), self.data['sentiment'].tolist(), test_size=self.test_size_val, random_state=self.random_state
        )
        return train_texts, test_texts, train_labels, test_labels

    def sample_data(self, data, n_samples_per_class):
        negatives = data[data['sentiment'] == 0]
        positives = data[data['sentiment'] == 1]

        sampled_negatives = negatives.sample(n=n_samples_per_class, random_state=self.random_state)
        sampled_positives = positives.sample(n=n_samples_per_class, random_state=self.random_state)

        sampled_data = pd.concat([sampled_negatives, sampled_positives])
        sampled_data = sampled_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        return sampled_data

    def tokenize_data(self, texts, max_length):
        print(f'[INFO]\tTokenizing data using \'{self.model_name}\'...')
        return self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_token_length)

    def save_split_data(self, file_path, train_texts, test_texts, train_labels, test_labels):
        with open(file_path, 'wb') as f:
            pickle.dump((train_texts, test_texts, train_labels, test_labels), f)

    def load_split_data(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f,)

if __name__ == "__main__":

    N=1000
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'raw', 'sentiment140.csv')

    print(f'[INFO]\tLoading {2*N} samples of data...')
    processor = DataProcessor(file_path, 
                              test_size=0.2, 
                              n_samples_per_class=N,
                              max_token_length=128)
    
    print('[INFO]\tSplitting data...')
    train_texts, test_texts, train_labels, test_labels = processor.preprocess_data()

    train_encodings = processor.tokenize_data(train_texts, max_length=128)
    test_encodings = processor.tokenize_data(test_texts, max_length=128)
    
    save_name = f'train_test_{processor.test_size()}_split.pkl'
    save_path = os.path.join(current_dir, '..', 'data', 'processed', save_name)
    processor.save_split_data(save_path, train_encodings, test_encodings, train_labels, test_labels)
    print(f'[INFO]\tProcessed data saved at {save_path}.')