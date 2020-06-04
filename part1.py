import numpy as numpy
import pandas as pd
from linreg import LinRegModel

def main():

    # read csv
    training_df = pd.read_csv('mpg_training.csv') # TODO do not hardcode path?

    model = LinRegModel()
    model.train(
        df=training_df,
        y_attr_name='mpg',
        iterations=10,
        learning_rate=1
    )

if __name__ == "__main__":
    main()