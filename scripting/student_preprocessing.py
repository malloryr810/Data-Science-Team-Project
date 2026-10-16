# Run: python scripting/student_preprocessing.py
# This script preprocesses the student dropout dataset for Machine Learning Pipelines.

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
OUT = ROOT / "outputs/week4"
OUT.mkdir(parents=True, exist_ok=True)
STUDENT_FILE = ROOT / "data/raw/student_dropout/data.csv"

raw_s = pd.read_csv(STUDENT_FILE, sep=";", encoding="utf-8-sig")
raw_s.columns = raw_s.columns.str.strip()

# 1. Filter out Enrolled records for binary classification
s = raw_s.loc[raw_s["Target"].isin(["Graduate", "Dropout"])].copy()

# 2. Separate Features and Target
X = s.drop(columns=["Target"])
y = s["Target"].map({"Graduate": 1, "Dropout": 0})  # Binary mapping

# 3. Stratified Train-Test Split (Prevents Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} rows")
print(f"Test set: {X_test.shape[0]:,} rows")

# 4. Identify column types for pipeline
# Nominal / Categorical columns that need encoding
nominal_cols = [
    "Marital status", "Application mode", "Course", "Previous qualification", 
    "Nacionality", "Mother's qualification", "Father's qualification", 
    "Mother's occupation", "Father's occupation"
]

# Binary categorical columns (already 0/1 or similar, but good to handle safely)
binary_cols = [
    "Daytime/evening attendance", "Displaced", "Educational special needs", 
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"
]

# Remaining columns are numeric (demographics, macroeconomics, academic units/grades)
numeric_cols = [col for col in X.columns if col not in nominal_cols + binary_cols]

# 5. Build Training-Fitted Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, nominal_cols + binary_cols)
    ]
)

# 6. Fit ONLY on Training Data, Transform Both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed training features shape: {X_train_processed.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")

# Save processed shapes summary
summary_df = pd.DataFrame({
    "Metric": ["Train Rows", "Test Rows", "Raw Features", "Processed Features"],
    "Value": [len(X_train), len(X_test), X.shape[1], X_train_processed.shape[1]]
})
summary_df.to_csv(OUT / "student_preprocessing_summary.csv", index=False)
print(f"\nSaved preprocessing summary to {OUT / 'student_preprocessing_summary.csv'}")