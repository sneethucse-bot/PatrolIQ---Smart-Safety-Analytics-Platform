import pandas as pd

def load_and_sample_data(csv_path, sample_size=500_000, random_state=42):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print(f"Original size: {len(df):,}")

    df = df.sample(n=sample_size, random_state=random_state)
    print(f"Sampled size: {len(df):,}")

    return df
