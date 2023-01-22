# MT-HF
A Machine Translation engine that utilizes a pre-trained model from HuggingFace, specifically the mBART-50 model.
# How to Use:
1. If you don't have conda installed on your machine, install it by following the instructions at this link: <br> https://docs.conda.io/projects/conda/en/latest/user-guide/install/<br>
2. Create a new environment and install the required packages by running the command:<br>
`conda create --name <env> --file requirements.txt`<br>
3. Set the parameters through config.yaml
4. Execute the python script `index.py`
# Inference
The `predict.py` script can be utilized to translate sentences with the trained model.
