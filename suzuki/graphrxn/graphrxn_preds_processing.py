'''
Processing the Graphrxn preds
'''

import pandas as pd
import numpy as np
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=str)
    return parser.parse_args()

def MAE(true, pred):
    return np.mean(np.abs(true - pred))

def find_pred_mae(df):
    mean = np.mean(df['origin_output'])
    std = np.std(df['origin_output'])
    true = np.array(df['origin_output'].tolist())
    preds = df['pred_0'].tolist()
    scaled_preds = [((x * std) + mean) for x in preds]
    mae = MAE(true, np.array(scaled_preds))
    return mae

def main():
    args = init_args()
    preds_path = args.preds_path

    df = pd.read_csv(preds_path)

    mae = find_pred_mae(df)

    print(f'The MAE for {preds_path} is {mae}.')

if __name__ == '__main__':  
    main()
