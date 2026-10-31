import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/cell2cell.csv")

# 1. Verify dropblk vs dropvce + blckvce
diff = np.abs(df["dropblk"] - (df["dropvce"] + df["blckvce"]))
exact_matches = int((diff < 1e-4).sum())
total_rows = len(df)
max_diff = float(diff.max())
print(f"=== Verification 1: dropblk vs (dropvce + blckvce) ===")
print(f"Total records: {total_rows:,}")
print(f"Exact matches (|diff| < 1e-4): {exact_matches:,} ({exact_matches/total_rows*100:.4f}%)")
print(f"Max absolute discrepancy: {max_diff}")
if exact_matches == total_rows:
    print("CONFIRMED: dropblk == dropvce + blckvce for 100.0% of records in the dataset.")

# 2. Verify retcall vs retcalls
print(f"\n=== Verification 2: retcall vs retcalls ===")
ct = pd.crosstab(df["retcall"], df["retcalls"], margins=True)
print("Cross-tabulation matrix:")
print(ct)

# Check if retcall is strictly the binary indicator of retcalls > 0
is_retcall_exact_indicator = ((df["retcall"] == 1) == (df["retcalls"] > 0)).all()
print(f"\nIs retcall strictly equivalent to (retcalls > 0)? {is_retcall_exact_indicator}")

corr_retcall = df["retcall"].corr(df["churn"])
corr_retcalls = df["retcalls"].corr(df["churn"])
print(f"Pearson r (retcall, churn):  {corr_retcall:.5f}")
print(f"Pearson r (retcalls, churn): {corr_retcalls:.5f}")
print(f"Pearson r (retcall, retcalls): {df['retcall'].corr(df['retcalls']):.5f}")

# Distribution of non-zero retcalls
print("\nDistribution of retcalls when > 0:")
print(df[df["retcalls"] > 0]["retcalls"].value_counts())
