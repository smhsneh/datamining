import pandas as pd
import os

def load_and_clean(filepath):
    print("loading dataset...")
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

    print(f"raw shape: {df.shape}")
    print(f"columns: {list(df.columns)}")

    # drop rows with missing customer id or description
    df.dropna(subset=["Customer ID", "Description"], inplace=True)

    # remove cancelled orders (invoice starts with 'C')
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # remove negative or zero quantities
    df = df[df["Quantity"] > 0]

    # remove negative prices
    df = df[df["Price"] > 0]

    # clean description
    df["Description"] = df["Description"].str.strip().str.upper()

    print(f"clean shape: {df.shape}")
    print(f"unique customers: {df['Customer ID'].nunique()}")
    print(f"unique products: {df['Description'].nunique()}")
    print(f"unique invoices: {df['Invoice'].nunique()}")

    return df


def build_basket(df, country="United Kingdom"):
    print(f"\nbuilding basket matrix for: {country}")
    df_country = df[df["Country"] == country]

    basket = (
        df_country.groupby(["Invoice", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket_encoded = basket.map(lambda x: 1 if x > 0 else 0)

    print(f"basket shape: {basket_encoded.shape}")
    return basket_encoded


if __name__ == "__main__":
    filepath = os.path.join("data", "online_retail.csv")
    df = load_and_clean(filepath)
    df.to_csv(os.path.join("data", "cleaned_retail.csv"), index=False)
    print("\nsaved cleaned data to data/cleaned_retail.csv")

    basket = build_basket(df)
    basket.to_csv(os.path.join("data", "basket_matrix.csv"))
    print("saved basket matrix to data/basket_matrix.csv")