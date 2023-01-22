from classes.main import Main
import csv
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch

class Inference(Main):
    def __init__(self):
        super().__init__()
        self.tokenizer.src_lang = self.cfg.params["src"]

    def translate(self, text):
        """
        Taking an input text and translate it into the target language
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        output = self.model.generate(input_ids,
            use_cache=True,
            max_length=100,
            attention_mask=attention_mask,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.cfg.params["tgt"]]
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    # Make an instance from the class
    I = Inference()
    # Read the path from config.yaml for prediction
    data = pd.read_csv(I.cfg.dataset["predict_path"], 
    header=None, quoting=csv.QUOTE_NONE, sep='\t', names=['src', 'ref']).dropna()
    print(data)
    # Apply the translate function to each src's tuple and 
    # Add the translated text to a new column named `mt``
    data['mt'] = data['src'].progress_apply(lambda x: I.translate(x))
    print(data)
    # Save the translation in a csv file
    data.to_csv(I.cfg.dataset["predict_path"]+"-trans.csv", index=False, quoting=csv.QUOTE_NONE,  sep='\t')


if __name__ == '__main__':
    main()
