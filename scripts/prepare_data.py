import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 1. File paths
# ==========================================

INPUT_FILE = "data/raw/gesture_data.csv"
OUTPUT_FOLDER = "data/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. Load dataset
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total samples: {len(df)}")

print("\nOriginal class distribution:")
print(df["gesture"].value_counts())


# ==========================================
# 3. Balance dataset
# ==========================================

MIN_SAMPLES = df["gesture"].value_counts().min()

print(f"\nUsing {MIN_SAMPLES} samples per gesture.")

balanced_data = []

for gesture in df["gesture"].unique():

    gesture_data = df[df["gesture"] == gesture]

    gesture_data = gesture_data.sample(
        n=MIN_SAMPLES,
        random_state=42
    )

    balanced_data.append(gesture_data)


df_balanced = pd.concat(
    balanced_data,
    ignore_index=True
)

df_balanced = df_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nBalanced class distribution:")
print(df_balanced["gesture"].value_counts())


# ==========================================
# 4. Separate features and labels
# ==========================================

X = df_balanced.drop("gesture", axis=1)
y = df_balanced["gesture"]


# ==========================================
# 5. Normalize landmarks relative to wrist
# ==========================================

print("\nNormalizing hand landmarks...")

normalized_features = []

for _, row in X.iterrows():

    landmarks = row.values.reshape(21, 3)

    # Wrist is landmark 0
    wrist = landmarks[0].copy()

    # Move wrist to origin
    landmarks = landmarks - wrist

    normalized_features.append(
        landmarks.flatten()
    )


X_normalized = pd.DataFrame(
    normalized_features,
    columns=X.columns
)

print(
    "Normalized feature shape:",
    X_normalized.shape
)


# ==========================================
# 6. Train-test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. Standard scaling
# ==========================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 8. Save processed data
# ==========================================

X_train_df = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

X_train_df.to_csv(
    f"{OUTPUT_FOLDER}/X_train.csv",
    index=False
)

X_test_df.to_csv(
    f"{OUTPUT_FOLDER}/X_test.csv",
    index=False
)

y_train.to_csv(
    f"{OUTPUT_FOLDER}/y_train.csv",
    index=False
)

y_test.to_csv(
    f"{OUTPUT_FOLDER}/y_test.csv",
    index=False
)


# ==========================================
# 9. Save scaler
# ==========================================

joblib.dump(
    scaler,
    f"{OUTPUT_FOLDER}/scaler.pkl"
)


# ==========================================
# 10. Done
# ==========================================

print("\n================================")
print(" DATA PREPARATION COMPLETE")
print("================================")

print("\nFiles created:")

print("✓ data/processed/X_train.csv")
print("✓ data/processed/X_test.csv")
print("✓ data/processed/y_train.csv")
print("✓ data/processed/y_test.csv")
print("✓ data/processed/scaler.pkl")