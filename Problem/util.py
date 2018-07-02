import pandas as pd


def verbose_print(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.head())