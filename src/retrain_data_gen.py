import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
n = 10000

# Randomly assign labels (CONFIRMED / CANDIDATE / FALSE POSITIVE)
labels = np.random.choice(
    ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"], 
    size=n, 
    p=[0.4, 0.4, 0.2]  # slightly more confirmed & candidate cases
)

# Generate synthetic feature data with realistic value ranges
data = {
    "disposition_score": np.clip(np.random.normal(0.7, 0.2, n), 0, 1),
    "ntl_fpflag": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    "se_fpflag": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    "co_fpflag": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    "ec_fpflag": np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
    "koi_period": np.random.uniform(0.5, 500, n),
    "koi_depth": np.random.uniform(50, 20000, n),
    "koi_prad": np.random.uniform(0.3, 20, n),
    "koi_eqtemp": np.random.uniform(100, 3000, n),
    "koi_stefftemp": np.random.uniform(3000, 10000, n),
    "koi_srad": np.random.uniform(0.1, 10, n),
    "koi_slogg": np.random.uniform(3.5, 5.0, n),
    "koi_kepmag": np.random.uniform(8, 16, n),
    "exoplanet_archive_disposition": labels
}

df = pd.DataFrame(data)

# Save CSV
output_file = "../data/kepler_retrain_10000.csv"
df.to_csv(output_file, index=False)

print(f"✅ Created '{output_file}' with {n} synthetic Kepler records!")
print(df.head())
