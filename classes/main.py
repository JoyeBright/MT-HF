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

    def load_dataset(self):
        """
        train, dev and test sets are loaded here
        NB: the supported format is csv and no need for a header
        """
        raw_datasets = load_dataset("csv", sep='\t', quoting=csv.QUOTE_NONE, data_files={
        "train": [self.cfg.dataset["train_path"]],
        "dev":   [self.cfg.dataset["dev_path"]],
        "test":  [self.cfg.dataset["test_path"]]})
        self.raw_datasets = raw_datasets


        