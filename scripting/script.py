# %%
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
relative_path = Path("data/raw/student_dropout/data.csv")
data_path = next(
    (folder / relative_path for folder in [project_root, *project_root.parents]
     if (folder / relative_path).is_file()),
    None,
)
if data_path is None:
    raise FileNotFoundError("Run this code from inside the project repository.")

students = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
students.columns = students.columns.str.strip()
print(f"Original dataset: {students.shape[0]:,} rows, {students.shape[1]} columns")
print(students.head().to_string())

# %%
binary_students = students.loc[students["Target"].isin(["Graduate", "Dropout"])].copy()
print(f"Removed {len(students) - len(binary_students):,} records outside the binary task.")
print(f"Binary dataset: {len(binary_students):,} rows")
print(f"Missing cells: {binary_students.isna().sum().sum()}")
print(f"Exact duplicate rows: {binary_students.duplicated().sum()}")

class_counts = binary_students["Target"].value_counts()
class_summary = pd.DataFrame({
    "Count": class_counts,
    "Percent": (class_counts / len(binary_students) * 100).round(2),
})
print(class_summary.to_string())

X = binary_students.drop(columns="Target")
y = binary_students["Target"]
