import os
import pandas as pd
from sklearn.model_selection import train_test_split

def list_parquet_files(folder_path):
    """List all .parquet files in the given folder."""
    parquet_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

def combine_parquet_files(parquet_files):
    """Combine all .parquet files into a single DataFrame."""
    df_list = [pd.read_parquet(file) for file in parquet_files]
    return pd.concat(df_list, ignore_index=True)

def split_dataframe(df, test_size=0.2, valid_size=0.5, random_state=42):
    """Split the DataFrame into train, test, and validation sets."""
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state)
    test_df, valid_df = train_test_split(temp_df, test_size=valid_size, random_state=random_state)
    return train_df, test_df, valid_df

def save_dataframes(train_df, test_df, valid_df, folder_path):
    """Save each split DataFrame as a .csv file."""
    train_df.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(folder_path, 'test.csv'), index=False)
    valid_df.to_csv(os.path.join(folder_path, 'valid.csv'), index=False)

def main():
    folder_path = '/Users/shrijayan.rajendran/projects/personal/monk/modern_bert-finetuning/data/raw/banking77/'
    parquet_files = list_parquet_files(folder_path)
    combined_df = combine_parquet_files(parquet_files)
    train_df, test_df, valid_df = split_dataframe(combined_df)
    save_dataframes(train_df, test_df, valid_df, folder_path)

if __name__ == "__main__":
    main()