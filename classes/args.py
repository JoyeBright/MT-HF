from transformers import Seq2SeqTrainingArguments
from classes.main import Main

class Args(Main):
    def __init__(self):
        # Access to the Main class
        super().__init__()
        # Training Arguments
        self.training_args = Seq2SeqTrainingArguments(
            self.PROJECT_NAME, 
            per_device_train_batch_size=self.cfg.params["device_train_bs"], 
            per_device_eval_batch_size=self.cfg.params["device_eval_bs"],
            num_train_epochs=self.cfg.params["epoch"],
            logging_steps=self.cfg.params["logging_step"],
            save_total_limit=self.cfg.params["save"],
            save_steps=self.cfg.params["save_steps"],
            weight_decay=self.cfg.params["weight_decay"],
            warmup_steps=self.cfg.params["warmup_steps"],
            seed=self.cfg.params["seed"],
            fp16=self.cfg.params["fp16"],
            push_to_hub=False,
            learning_rate=float(self.cfg.params["lr"]),
            evaluation_strategy= self.cfg.params["evaluation"],
            eval_steps=self.cfg.params["eval_steps"], # Evaluation and Save happens every n steps
            load_best_model_at_end= self.cfg.params["best_model"],
            metric_for_best_model= self.cfg.params["metric_best_model"],
            do_train=self.cfg.MT["do_train"],
            do_eval=self.cfg.MT["do_eval"],
            do_predict=self.cfg.MT["do_predict"], 
            optim=self.cfg.params["optim"],
            predict_with_generate=self.cfg.MT["predict_with_generate"],
        )