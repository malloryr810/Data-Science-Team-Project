# Run with: python scripting/regression_data_check.py
# (requires pandas, openpyxl, matplotlib, and seaborn).
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

# %%
missing_summary = pd.DataFrame({
    "Missing count": X.isna().sum(),
    "Missing percent": (X.isna().mean() * 100).round(2),
}).sort_values("Missing count", ascending=False)
print("\nMissing values in regression predictors:")
print(missing_summary.to_string())

# %%
# Tukey's fences flag potential outliers for investigation; nothing is removed here.
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
below_lower = y < lower_fence
above_upper = y > upper_fence
potential_outliers = below_lower | above_upper

print("\nUTS Tukey's fences (MPa):")
print(f"Q1 (25th percentile): {q1:.2f}")
print(f"Q3 (75th percentile): {q3:.2f}")
print(f"IQR (Q3 - Q1): {iqr:.2f}")
print(f"Lower fence (Q1 - 1.5 x IQR): {lower_fence:.2f}")
print(f"Upper fence (Q3 + 1.5 x IQR): {upper_fence:.2f}")
print(f"Potential outliers: {potential_outliers.sum():,} "
      f"({potential_outliers.mean() * 100:.2f}% of {len(y):,} UTS observations)")
print(f"Below lower fence: {below_lower.sum():,}")
print(f"Above upper fence: {above_upper.sum():,}")
print(
    "Interpretation: Tukey's fences only identify observations to investigate. "
    "The team should check whether each one is a data-entry error or a valid "
    "extreme measurement before deciding how to handle it. No values were removed or changed."
)

output_dir = project_root / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
fig, (hist_ax, box_ax) = plt.subplots(
    2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 1]}
)
sns.histplot(x=y, bins=30, ax=hist_ax)
# Both fences fall inside the observed UTS range, so they fit without distorting the axis.
for fence, name in [(lower_fence, "Lower fence"), (upper_fence, "Upper fence")]:
    hist_ax.axvline(fence, color="tab:red", linestyle="--")
    hist_ax.text(fence + 15, 0.97, f"{name}\n{fence:.0f} MPa", color="tab:red",
                 ha="left", va="top", fontsize=9, transform=hist_ax.get_xaxis_transform())
hist_ax.set_title("Distribution of Ultimate Tensile Strength")
hist_ax.set_xlabel("Ultimate Tensile Strength (MPa)")
hist_ax.set_ylabel("Number of specimens")
sns.boxplot(x=y, ax=box_ax)
box_ax.set_title("Box plot of Ultimate Tensile Strength")
box_ax.set_xlabel("Ultimate Tensile Strength (MPa)")
box_ax.set_xlim(hist_ax.get_xlim())
fig.tight_layout()
figure_path = output_dir / "uts_distribution_and_outliers.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)
print(f"\nSaved UTS distribution figure to: {figure_path}")
