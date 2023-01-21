import numpy as np
import evaluate
from classes.main import Main

class Eval(Main):
    def __init__(self):
        super().__init__()

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metric(self, eval_preds):

        preds, labels = eval_preds
        print(len(preds))
        preds = preds[0] if isinstance(preds, tuple) else preds
        print(len(preds))
        pred_str =self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        print(pred_str)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(label_str)
        
        # Some simple post-processing
        pred_str, label_str = self.postprocess_text(pred_str, label_str)
        
        sacrebleu = evaluate.load("sacrebleu")

        result = sacrebleu.compute(predictions=pred_str, references=label_str)
        print(list(result.keys()))
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result