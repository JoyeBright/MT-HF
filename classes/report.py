from transformers import Seq2SeqTrainer
from classes.main import Main
from typing import Dict

class MySeq2SeqTrainer(Seq2SeqTrainer, Main):
    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)
        
        f = open(self.PROJECT_NAME + ".txt", "a")
        f.write(str(logs) + '\n' )
        f.close()