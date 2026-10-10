# Run with: python scripting/regression_data_check.py (requires pandas and openpyxl).
# %%
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root / "data/raw/tensile_strength"
    / "Tensile_Properties_of_Sub-sized_Specimens.xlsx"
)

with pd.ExcelFile(data_path) as workbook:
    print(f"Workbook sheets: {workbook.sheet_names}")
    # The first row groups columns; the second row contains the actual names.
    materials = pd.read_excel(workbook, sheet_name="Tensile Data", header=1)

materials.columns = materials.columns.str.strip()
target = "Ultimate Tensile Strength (MPa)"
print(f"Original dataset: {materials.shape[0]:,} rows, {materials.shape[1]} columns")
print(f"Columns: {materials.columns.tolist()}")
print(materials[["Material", "Test Temperature (C)", target]].head().to_string(index=False))
print(f"Missing cells in original dataset: {materials.isna().sum().sum():,}")
print(f"Exact duplicate rows in original dataset: {materials.duplicated().sum():,}")

# %%
# A missing target cannot be used for supervised regression; keep raw data intact.
regression_materials = materials.loc[materials[target].notna()].copy()
print(f"Rows without UTS: {materials[target].isna().sum():,}")
print(f"Rows with recorded UTS: {len(regression_materials):,}")
print("\nUTS summary (MPa):")
print(regression_materials[target].describe().round(2).to_string())

# Other tensile outcomes leak experimental results. References are audit metadata.
excluded_columns = [
    target,
    "Yield Strength (MPa)",
    "Uniform Elongation (%)",
    "Total Elongation (%)",
    "Reference",
]
X = regression_materials.drop(columns=excluded_columns)
y = regression_materials[target]
print(f"\nInitial predictor table: {X.shape[0]:,} rows, {X.shape[1]} columns")
