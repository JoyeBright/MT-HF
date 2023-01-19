import pandas as pd
import csv
from classes.main import Main
from sklearn.model_selection import train_test_split

class Splitter(Main):
    def __init__(self):
        # Access to the Main class
        super().__init__()
        # Call read_text
        X, y = self.read_txt()
        # Split the text file
        X_train, X_test, y_train, y_test = self.split(X, y)
        # The 2nd split to create dev set using the new train set
        X_train, X_dev, y_train, y_dev = self.split(X_train, y_train)
        print("Train length:", len(X_train), len(y_train))
        print("Dev length:",   len(X_dev),   len(y_dev))
        print("Test length:",  len(X_test),  len(y_test))
    def read_txt(self):
        """
        Reads the text file and load it in a dataframe
        Then returns the source and target of the text file
        """
        data = pd.read_csv(self.cfg.splitter['path'], 
                        header=None, quoting=csv.QUOTE_NONE,
                        sep='\t', names=['src', 'tgt']
                        ).dropna()
        print(data)
        X = data['src'].values
        y = data['tgt'].values
        return X, y
    
    def split(self, X, y):
        """
        Take X and y and split them into two sets 
        Given the proportion of the dataset to include in the test/dev split
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.splitter['split'],
            random_state=self.cfg.params['seed'], shuffle=True)
        return X_train, X_test, y_train, y_test

