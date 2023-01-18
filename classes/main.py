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
    


        