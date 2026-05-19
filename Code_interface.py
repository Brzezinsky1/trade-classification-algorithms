import pandas as pd
pd.set_option('display.max_columns',None)

def classify_side(trades: pd.DataFrame) -> pd.Series:
    """
    input is a pandas DataFrame with trade price and volume columns (time-indexed); 
    output is a boolean pandas Series aligned to the input where True = sell aggressor and False = buy aggressor (matching the convention of the sample data).
    """
    predicted_sides = []

    predictions = pd.Series( predicted_sides , index = trades.index ) #pandas Series aligned to the input by index = trades.index

    return predictions


