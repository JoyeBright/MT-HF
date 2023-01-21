from transformers import (DataCollatorForSeq2Seq,
                          MBart50TokenizerFast,
                          MBartForConditionalGeneration,
                          set_seed
                         )
from datasets import load_dataset
from torch.utils.data import DataLoader
import csv
import yaml
from munch import Munch
import os

class Main:
    def __init__(self):
        # Loading the configuration file in cfg
        with open("config.yaml", "r") as file:
            cfg = yaml.safe_load(file)
        # Converting dictionary to object
        self.cfg = Munch(cfg)
        # set some params
        set_seed(self.cfg.params["seed"])
        # MLflow setup
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.cfg.mlflow["exp_name"]
        os.environ["MLFLOW_FLATTEN_PARAMS"] = self.cfg.mlflow["params"]
        self.PROJECT_NAME = self.cfg.mlflow["exp_name"] + self.cfg.mlflow["params"]
        # Defining the tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.cfg.params["checkpoint"],
            src_lang=self.cfg.params["src"],
            tgt_lang=self.cfg.params["tgt"], 
            do_lower_case=self.cfg.params["lower_case"],
            normalization=self.cfg.params["normalization"]
        )
        # Define the model
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.cfg.params["checkpoint"])

    def load_dataset(self):
        """
        train, dev and test sets are loaded here
        CSV and '\t' are the supported format and delimiter, respectively. 
        Inputs need to have two columns labeled with 'src' and 'tgt'.
        """
        raw_datasets = load_dataset("csv", sep='\t', quoting=csv.QUOTE_NONE, data_files={
        "train": [self.cfg.dataset["train_path"]],
        "dev":   [self.cfg.dataset["dev_path"]],
        "test":  [self.cfg.dataset["test_path"]]})
        self.raw_datasets = raw_datasets
        return raw_datasets
    
    def tokenize_function(self, example):
        """
        Taking src and tgt and tokenize them using the initialized tokenizer
        """
        return self.tokenizer(example["src"], text_target=example["tgt"],
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
    


        