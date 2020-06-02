import numpy as numpy
import pandas as pd
from linreg import LinReg

def main():

    #read csv
    training_df = pd.read_csv('forestfires_training.csv') # TODO do not hardcode path

    model = LinReg(training_df, 1, 1)
    model.train()

if __name__ == "__main__":
    main()