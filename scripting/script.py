# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

class_counts = binary_students["Target"].value_counts().reindex(["Graduate", "Dropout"], fill_value=0)
class_summary = pd.DataFrame({
    "Count": class_counts,
    "Percent": (class_counts / len(binary_students) * 100).round(2),
})
print("\nClass distribution:")
print(class_summary.to_string(formatters={"Count": "{:,.0f}".format, "Percent": "{:.2f}%".format}))

output_dir = data_path.parents[3] / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots()
ax.bar(class_counts.index, class_counts.values)
ax.set_title("Student outcomes: Graduate vs. Dropout")
ax.set_xlabel("Class")
ax.set_ylabel("Number of students")
fig.tight_layout()
fig.savefig(output_dir / "class_distribution.png", dpi=150)
plt.close(fig)

X = binary_students.drop(columns="Target")
y = binary_students["Target"]


sem1_cols = [c for c in X.columns if "1st sem" in c.lower()]
sem2_cols = [c for c in X.columns if "2nd sem" in c.lower()]

print(f"\n1st Semester features ({len(sem1_cols)}): {sem1_cols}")
print(f"2nd Semester features ({len(sem2_cols)}): {sem2_cols}")

# both academic cols for correlation analysis
academic_cols = sem1_cols + sem2_cols
corr_matrix = X[academic_cols].corr()

print("\nCorrelation matrix preview (Academic Variables):")
print(corr_matrix.iloc[:5, :5].round(2).to_string())

# heatmap
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(cax, label="Pearson Correlation")

# ticks and labels
ax.set_xticks(np.arange(len(academic_cols)))
ax.set_yticks(np.arange(len(academic_cols)))
ax.set_xticklabels(academic_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(academic_cols, fontsize=8)

# cell annotations
for i in range(len(academic_cols)):
    for j in range(len(academic_cols)):
        val = corr_matrix.iloc[i, j]
        text_color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

ax.set_title("Correlation Heatmap: 1st vs. 2nd Semester Academic Variables")
fig.tight_layout()
fig.savefig(output_dir / "semester_correlation_heatmap.png", dpi=150)
plt.close(fig)
print(f"\nSaved correlation heatmap to: {output_dir / 'semester_correlation_heatmap.png'}")
