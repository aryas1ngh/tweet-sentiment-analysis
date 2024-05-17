from transformers import Trainer, TrainingArguments, TrainerCallback
from model import get_model, TweetDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pickle, torch
import sys,os
import copy
sys.path.append('../')

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
        

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def load_split_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'processed', 'train_test_0.2_split.pkl')

    print(f'[INFO]\tExtracting data from {file_path}...')
    train_texts, test_texts, train_labels, test_labels = load_split_data(file_path)
    # print(test_texts, test_labels)
    train_dataset = TweetDataset(train_texts, train_labels)
    test_dataset = TweetDataset(test_texts, test_labels)


    print('[INFO]\tInitializing model...')
    model = get_model()
    training_args = TrainingArguments(
        output_dir='../results/models',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='../results/logs',
        logging_steps=20,
        evaluation_strategy="epoch",
    )
    
    print('[INFO]\tTraining model...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(CustomCallback(trainer))
    trainer.train()


    print('[INFO]\tSaving model...')
    save_path = os.path.join(current_dir, '..', 'results', 'models', 'bert-finetuned')
    trainer.save_model(save_path)
    print(f'[INFO]\tModel saved at {save_path}.')

    train_dict={}
    for i in trainer.state.log_history:
        train_dict[i['step']]=i

    train_loss_file = f'train_loss_{training_args.num_train_epochs}_epochs.pt'
    torch.save(train_dict, os.path.join(current_dir, '..', 'results', 'logs', train_loss_file))

    print('[INFO]\tEvaluating model...')
    trainer.evaluate()
    print('[INFO]\tDone.')

if __name__ == "__main__":

    train()
