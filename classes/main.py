from transformers import (set_seed,
                          Seq2SeqTrainingArguments, 
                          EarlyStoppingCallback,
                          Trainer,
                          Seq2SeqTrainer,
                          logging,
                          DataCollatorForSeq2Seq,
                          MBartForConditionalGeneration,
                          MBart50TokenizerFast
                         )
from datasets import load_dataset
from torch.utils.data import DataLoader
import csv

class main():
    def __init__(self, cfg):
        # Loading the configuration file into cfg
        self.cfg = cfg
        # Defining the tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.cfg.params["checkpoint"],
            src_lang=self.cfg.params["src"],
            tgt_lang=self.cfg.params["tgt"], 
            do_lower_case=self.cfg.params["lower_case"],
            normalization=self.cfg.params["normalization"]
        )
        # Apply tokenization function to the sample
        self.tokenized_datasets = self.raw_datasets.map(
            self.tokenize_function, batched=True)
        # Remove the header (src and tgt) from the input
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(['src', 'tgt'])
        # Set the format: torch
        self.tokenized_datasets.set_format("torch")
        # call the data collator
        self.data_collator = self.data_collator(self.tokenizer)
        # call the data loader
        train_dataloader, dev_dataloader, test_dataloader = self.data_loader(self.tokenized_datasets, self.data_collator)


    def load_dataset(self):
        """
        train, dev and test sets are loaded here
        The supported format and delimiter are CSV and '\t', respectively. 
        Inputs need to have two columns that are labeled with 'src' and 'tgt'.
        """
        raw_datasets = load_dataset("csv", sep='\t', quoting=csv.QUOTE_NONE, data_files={
        "train": [self.cfg.dataset["train_path"]],
        "dev":   [self.cfg.dataset["dev_path"]],
        "test":  [self.cfg.dataset["test_path"]]})
        self.raw_datasets = raw_datasets
    
    def tokenize_function(self, example):
        """
        Taking src and tgt and tokenize them using the initialized tokenizer
        """
        return self.tokenizer(example["src"], example["tgt"],
        truncation=True, max_length=self.cfg.params["max_len"])
    
    def data_collator(self, tokenizer):
        """
         Forming batches by applying padding based on the max_length
         NB: max_length can be set in the config file.
        """
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        return data_collator

    def data_loader(self, tokenized_datasets, data_collator):
        """
        Create batches from train, dev, and test sets
        Batch size can be set in the config file.
        """
        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True, batch_size=self.cfg.params["train_bs"],
            collate_fn=data_collator
        )
        dev_dataloader = DataLoader(
            tokenized_datasets["dev"],
            batch_size=self.cfg.params["dev_bs"],
            collate_fn=data_collator
        )
        test_dataloader = DataLoader(
            tokenized_datasets["test"],
            batch_size=self.cfg.params["test_bs"],
            collate_fn=data_collator
        )
        return [train_dataloader, dev_dataloader, test_dataloader]
    


        