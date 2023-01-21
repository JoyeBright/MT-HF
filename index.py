from classes.main import Main
from classes.splitter import Splitter
from classes.report import MySeq2SeqTrainer
from classes.args import Args
from classes.eval import Eval
from transformers import EarlyStoppingCallback
import evaluate
import torch
import numpy as np
# Disable tokenizer_parallelism error
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Creating an instance (named run) from the main class
    run = Main()
    # Split the text file into train, dev and test sets
    Splitter() 
    # Call load_dataset
    raw_datasets = run.load_dataset()
    # Apply tokenization function to the sample
    tokenized_datasets = raw_datasets.map(
        run.tokenize_function, batched=True)
    # Remove the header (src and tgt) from the input
    tokenized_datasets = tokenized_datasets.remove_columns(['src', 'tgt'])
    # Set the format: torch
    tokenized_datasets.set_format("torch")
    # Call the data collator
    data_collator = run.data_collator(run.tokenizer)
    # Call the data loader
    train_dataloader, dev_dataloader, test_dataloader = run.data_loader(
        tokenized_datasets, run.data_collator)
    # Intialize the model
    model = run.model
    # Load the model in GPU (if available)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print("You're using:", device)
    # Call the training arguments
    training_args = Args().training_args
    # Define the trainer (derived from MySeq2SeqTrainer)
    trainer = MySeq2SeqTrainer(
        model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=run.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=run.cfg.params["early_stop"])],
        compute_metrics=Eval.compute_metric
    )
    # Train
    if run.cfg.params["do_train"]==True:
        trainer.train()
    
if __name__ == '__main__':
    main()

