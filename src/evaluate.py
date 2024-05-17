import torch
from transformers import Trainer, TrainingArguments, BertTokenizer
from model import get_model, TweetDataset
from data_processing import DataProcessor
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def evaluate():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_proc = DataProcessor(current_dir+'/../data/raw/sentiment140.csv')

    model = get_model()
    model_path = os.path.join(current_dir, '..', 'results', 'models', 'bert-finetuned', 'pytorch_model.bin')
    model.load_state_dict(torch.load(model_path))

    _, test_texts, _, test_labels = data_proc.load_split_data(current_dir+'/../data/processed/train_test_0.2_split.pkl')
    eval_data = TweetDataset(test_texts, test_labels)

    training_args_path = os.path.join(current_dir, '..', 'results', 'logs')
    training_args = TrainingArguments(output_dir=training_args_path, per_device_eval_batch_size=16)
    trainer = Trainer(model=model, args=training_args, eval_dataset=eval_data)
    results = trainer.evaluate()
    print("[INFO]\tEvaluation Results:")
    print('\t',results)

    train_loss = torch.load(current_dir+'/../results/logs/train_loss_10_epochs.pt')
    step_losses = []
    step_LRs = []
    eval_dict = {
        'eval_loss': [],
        'eval_accuracy': [],
        'eval_precision': [],
        'eval_recall': [],
        'eval_f1': []
    }

    for k,v in train_loss.items():
        # print('Key: {}, Value: {}'.format(k,v))
        if 'loss' in v.keys(): 
            step_losses.append(v['loss'])
            step_LRs.append(v['learning_rate']) # recorded at every 10th step, contains 90 entries
        if 'eval_loss' in v.keys(): 
             # recorded at every epoch, has 9 entries
            eval_dict['eval_loss'].append(v['eval_loss'])
            eval_dict['eval_accuracy'].append(v['eval_accuracy'])
            eval_dict['eval_precision'].append(v['eval_precision'])
            eval_dict['eval_recall'].append(v['eval_recall'])
            eval_dict['eval_f1'].append(v['eval_f1'])

    lr_scaling_factor = max(step_losses) / max(step_LRs)
    print('[INFO]\tPlotting learning rate graph with scaling factor: {}'.format(lr_scaling_factor))
    scaled_step_LRs = [lr * lr_scaling_factor for lr in step_LRs]

    plt.figure(figsize=(10, 6))
    plt.plot(step_losses, label='Training Loss', color='tab:red')
    plt.plot(scaled_step_LRs, label='Learning Rate (scaled x1e-05)', color='tab:blue')

    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Training Loss and Learning Rate')
    plt.legend()
    plt.savefig(current_dir+'/../results/plots/train_loss_lr.png')


    epochs = range(1, len(eval_dict['eval_loss']) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, eval_dict['eval_accuracy'], marker='o', label='Eval Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, eval_dict['eval_precision'], marker='o', label='Eval Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Evaluation Precision')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, eval_dict['eval_recall'], marker='o', label='Eval Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Evaluation Recall')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, eval_dict['eval_f1'], marker='o', label='Eval F1')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Evaluation F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(current_dir+'/../results/plots/eval_metrics.png')

if __name__ == '__main__':
    evaluate()