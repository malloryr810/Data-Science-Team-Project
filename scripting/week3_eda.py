"""Review data without changing source files or training models.

Run: python scripting/week3_eda.py
"""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
# Save figures without opening a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/week3"
STUDENT_FILE = ROOT / "data/raw/student_dropout/data.csv"
TENSILE_FILE = ROOT / "data/raw/tensile_strength/Tensile_Properties_of_Sub-sized_Specimens.xlsx"
UTS = "Ultimate Tensile Strength (MPa)"
OUTCOMES = {UTS, "Yield Strength (MPa)", "Uniform Elongation (%)", "Total Elongation (%)"}
BINARY = {"Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
          "Tuition fees up to date", "Gender", "Scholarship holder", "International"}
NOMINAL = {"Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
           "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"}
PROCESS = {"Manufacturing", "Specimen Preparation", "Treatment", "Treatment – Three Categories",
           "Cooling", "Microstructure", "Specimen Size Type"}
RATIOS = {"Aspect Ratio (L/W)": "Gauge Length (mm)", "Aspect Ratio (R/W)": "Fillet Radius (mm)",
          "Aspect Ratio (T/W)": "Thickness (mm)"}
BLUE, ORANGE = "#386a9c", "#bd662d"


def save_table(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False)


def profile(series):
    return {"missing_count": int(series.isna().sum()),
            "missing_percent": round(series.isna().mean() * 100, 2),
            "zero_count": int(series.eq(0).sum()), "distinct_count": int(series.nunique())}


def student_decisions(raw, task, metadata):
    definitions = {v["name"].strip().lower(): v for v in metadata["variables"]}
    rows = []
    for col in raw:
        definition = definitions[col.lower()]
        typ = "categorical" if col in NOMINAL else "binary categorical" if col in BINARY else "numeric"
        timing, dependency, unknown = "enrollment-related; verify collection snapshot", "", None
        meaning = definition["description"]
        missing = "No blanks in this snapshot; a valid zero must not be recoded as missing."
        action = "Retain; scale numeric inputs for scale-sensitive models using training data only."
        if col in NOMINAL | BINARY:
            action = "Retain as categorical; encode using a training-fitted encoder with unseen-category handling."
            missing = "No blanks; numeric codes are categories, not quantities to median-fill."
        if col in {"Mother's qualification", "Father's qualification"}:
            unknown = 34
            missing = "Code 34 is documented Unknown; preserve as an explicit unknown category."
        if col in {"Mother's occupation", "Father's occupation"}:
            unknown = 99
            missing = "Code 99 is documented blank; encode Unknown. Code 0 means Student and is valid."
        if " sem " in col:
            semester = "first" if "1st" in col else "second"
            timing = f"{semester}-semester record; individual observation/outcome times unavailable"
            meaning = f"{semester.capitalize()}-semester " + col.split("(", 1)[1].rstrip(")")
            dependency = "Compare paired semester fields; approved/enrolled/grade/evaluation counts overlap."
            if "(grade)" in col:
                typ = "numeric continuous"
                missing = "Zero coincides with no approved units in this snapshot; keep zero pending source confirmation."
                action = "Retain; investigate a no-approved-units indicator without treating zero as a median grade."
            else:
                typ = "numeric count"
                missing = "Zero is a count, not automatically missing; check consistency with related counts."
        if col in {"Debtor", "Tuition fees up to date", "Scholarship holder"}:
            timing = "administrative/financial status; snapshot date not supplied"
            dependency = "Financial statuses may overlap; check timing and categorical associations."
        if col in {"Unemployment rate", "Inflation rate", "GDP"}:
            timing = "macroeconomic context; year/window mapping not supplied"
            missing = "No blanks. Negative inflation/GDP values can be valid; do not replace them with zero."
        if col == "Application order":
            typ = "ordinal numeric"
            missing = "No blanks; 0 is documented first choice, not missing."
        if col == "Target":
            typ, timing = "target categorical", "final recorded course outcome"
            action = "Keep Graduate/Dropout only; exclude from predictors and never impute a target."
            missing = "No missing labels; Enrolled is excluded by the assignment, not a negative class."
        rows.append({"column": col, "role": "target" if col == "Target" else "allowed predictor",
                     "semantic_type": typ, "meaning": meaning, "availability": timing,
                     "raw_missing_count": int(raw[col].isna().sum()), **profile(task[col]),
                     "coded_unknown_count": int(task[col].eq(unknown).sum()) if unknown is not None else 0,
                     "missing_zero_meaning": missing, "proposed_action": action,
                     "dependency_review": dependency,
                     "status": "proposed; no transformation applied"})
    return pd.DataFrame(rows)


def tensile_decisions(raw, task, descriptions):
    # Fix spelling differences between the workbook sheets.
    descriptions = {str(row.iloc[0]).strip().replace("O (wt,%)", "O (wt.,%)")
                    .replace("Treatment - Three Categories", "Treatment – Three Categories"):
                    str(row.iloc[1]) for _, row in descriptions.iterrows()}
    rows = []
    for col in raw:
        role, typ = "allowed predictor", "numeric"
        timing, dependency = "pre-test condition/property; verify measurement provenance", ""
        missing = "Unreported measurement; neither zero nor an expected value is established by a blank."
        action = "Validate units; if retained, fit median plus missing-indicator strategy on training data only."
        if "(wt" in col:
            typ = "numeric composition with limits/ranges/balance markers"
            missing = "Blank does not establish elemental absence. Sparse reported composition is not a complete recipe."
            action = "Parse units/limits/ranges with uncertainty retained; do not blanket-fill zero or global median."
            dependency = "Compositions are constrained; calculate balance only with sufficiently complete known components."
        if col in PROCESS | {"Material", "Material Group"}:
            typ = "categorical"
            missing = "Unreported category; Unknown is distinct from confirmed None/not applicable."
            action = "Normalize formatting; review synonyms; encode Unknown and unseen categories explicitly."
        if col in {"Material", "Material Group"}:
            dependency = "Material and material group may overlap; inspect mapping before removing either."
        if col in {"Treatment", "Treatment – Three Categories"}:
            dependency = "Detailed and grouped treatment overlap; same normalized label can map to different groups."
        if col.startswith("Post Treatment"):
            missing = "Could be unreported or not applicable; condition on confirmed treatment, not an assumed zero."
            dependency = "Interpret together with treatment and cooling; a blank treatment does not prove no treatment."
        if col.startswith("Irradiation"):
            missing = "Could be unreported or not applicable; a blank does not prove unirradiated."
            action = "Retain known values; record reporting indicator, never label it irradiated without evidence."
            dependency = "Review dose/temperature/time jointly; 0 dose is observed and distinct from missing dose."
        if col in RATIOS:
            action = "Derive missing ratio only from validated numerator and positive width; investigate conflicts first."
            dependency = f"{RATIOS[col]} / Width (mm); avoid imputing dimensions and dependent ratios independently."
        if col in {"Gauge Length (mm)", "Thickness (mm)", "Width (mm)", "Fillet Radius (mm)"}:
            dependency = "Specimen geometry and ratios overlap; choose a coherent model representation."
            action = "Validate positive dimensions and specimen geometry; no automatic correction or zero fill."
        if col == "Test Temperature (C)":
            missing = "Unreported test temperature; 0 or negative Celsius can be valid and does not mean missing."
        if col == "Strain Rate (1/s)":
            action = "Check positivity/units; consider a log transform based on training data; no transform applied here."
        if col == "Specimen Size Type":
            dependency = "Labels vary by study; one observed label resembles material metadata rather than a size type."
        if col in OUTCOMES:
            role, timing = ("target" if col == UTS else "forbidden predictor"), "measured tensile-test outcome"
            missing = "Never impute the supervised target; other tensile outcomes are not predictor candidates."
            action = "Exclude rows without UTS." if col == UTS else "Exclude from every predictor table."
        if col == "Reference":
            role, typ, timing = "group/audit metadata; forbidden predictor", "categorical identifier", "source article identifier"
            missing = "No missing references in the observed-UTS subset."
            action = "Preserve for provenance and grouped evaluation; exclude from predictive inputs."
        nonnumeric = (task[col].notna() & pd.to_numeric(task[col], errors="coerce").isna()).sum() if typ.startswith("numeric") else 0
        rows.append({"column": col, "role": role, "semantic_type": typ,
                     "meaning": descriptions[col], "availability": timing,
                     "raw_missing_count": int(raw[col].isna().sum()), **profile(task[col]),
                     "nonnumeric_observed_count": int(nonnumeric), "missing_zero_meaning": missing,
                     "proposed_action": action, "dependency_review": dependency,
                     "status": "proposed; source confirmation needed where noted"})
    return pd.DataFrame(rows)


def correlations(frame, columns, stem):
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    # Count rows where both features are recorded.
    counts = numeric.notna().astype(int).T @ numeric.notna().astype(int)
    corr.to_csv(OUT / f"{stem}_correlations.csv")
    counts.to_csv(OUT / f"{stem}_pair_counts.csv")
    pairs = [{"feature_a": a, "feature_b": b, "pearson_r": corr.loc[a, b], "paired_rows": int(counts.loc[a, b])}
             for i, a in enumerate(columns) for b in columns[i + 1:] if pd.notna(corr.loc[a, b])]
    pairs = pd.DataFrame(pairs).sort_values("pearson_r", key=lambda x: x.abs(), ascending=False)
    save_table(pairs, f"{stem}_correlation_pairs")
    return corr, pairs


def plot_heatmap(ax, matrix, labels, title):
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title, pad=14)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix.iloc[i, j]
            ax.text(j, i, f"{val:.2f}" if pd.notna(val) else "NA", ha="center", va="center",
                    color="white" if pd.notna(val) and abs(val) > .55 else "#222222", fontsize=9)
    return im


def figures(students, tensile, academic_corr, tensile_corr, missing, zeros):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    counts = students.Target.value_counts().reindex(["Graduate", "Dropout"])
    bars = axes[0].bar(counts.index, counts, color=[BLUE, ORANGE])
    axes[0].bar_label(bars, [f"{v:,}\n{v / len(students):.1%}" for v in counts], padding=4)
    axes[0].set_ylim(0, counts.max() * 1.23)
    axes[0].set_ylabel("Students")
    axes[0].set_title(f"Binary class distribution (n={len(students):,})")
    for i, label in enumerate(["Graduate", "Dropout"]):
        values = zeros.loc[zeros.target.eq(label)].sort_values("semester")["zero_grade_percent"]
        axes[1].bar(np.arange(2) + (i - .5) * .32, values, width=.32, label=label,
                    color=[BLUE, ORANGE][i])
    axes[1].set_xticks([0, 1], ["First semester", "Second semester"])
    axes[1].set_ylabel("Zero-grade records within each class (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].set_title("Zero-grade recording by final outcome\nAssociation does not establish timing or causation")
    fig.savefig(OUT / "student_outcomes_and_zero_grades.png"); plt.close(fig)

    labels = [c.replace("Curricular units ", "").replace(" sem ", " ") for c in academic_corr]
    fig, ax = plt.subplots(figsize=(12, 11), layout="constrained")
    im = plot_heatmap(ax, academic_corr, labels, f"Semester academic correlations (n={len(students):,})")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=.7)
    fig.savefig(OUT / "student_semester_correlations.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    largest = missing.nlargest(18, "missing_percent").sort_values("missing_percent")
    bars = ax.barh(largest.column, largest.missing_percent, color=BLUE)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlim(0, 115); ax.set_xlabel("Missing predictor values (%)")
    ax.set_title(f"Most sparse UTS predictors (18 shown; n={len(tensile)})\nComplete column inventory is in tensile_column_decisions.csv")
    fig.savefig(OUT / "tensile_predictor_missingness.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    axes[0].hist(tensile[UTS], bins=30, color=BLUE, edgecolor="white")
    axes[0].set_xlabel("UTS (MPa)"); axes[0].set_ylabel("Specimens")
    axes[0].set_title(f"Observed UTS distribution (n={len(tensile)})")
    groups = sorted(tensile["Material Group"].dropna().unique(), key=lambda g: tensile.loc[tensile["Material Group"].eq(g), UTS].median())
    values = [tensile.loc[tensile["Material Group"].eq(g), UTS] for g in groups]
    axes[1].boxplot(values, orientation="horizontal", tick_labels=[f"{g} (n={len(v)})" for g, v in zip(groups, values)],
                    flierprops={"marker": ".", "markeredgecolor": ORANGE})
    axes[1].set_xlabel("UTS (MPa)"); axes[1].set_title("UTS by material group\nExtreme values require material-specific interpretation")
    fig.savefig(OUT / "tensile_uts_by_material.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    labels = [c.replace("Ultimate Tensile Strength", "UTS").replace("Test Temperature", "Test temp.").replace("Gauge Length", "Length") for c in tensile_corr]
    im = plot_heatmap(ax, tensile_corr, labels, "Tensile numeric relationships\nPairwise available observations; sample counts in companion CSV")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=.7)
    fig.savefig(OUT / "tensile_numeric_correlations.png"); plt.close(fig)

    selected = ["C (wt.%)", "Cr (wt.%)", "Ni (wt.%)", "Manufacturing", "Treatment",
                "Grain size (um)", "Irradiation Dose (mdpa)", "Gauge Length (mm)",
                "Width (mm)", "Test Temperature (C)", "Strain Rate (1/s)"]
    by_source = tensile[selected].isna().groupby(tensile.Reference).mean() * 100
    by_source.to_csv(OUT / "tensile_missingness_by_reference.csv")
    sizes = tensile.Reference.value_counts()
    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    im = ax.imshow(by_source, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(selected)), selected, rotation=45, ha="right")
    ax.set_yticks(range(len(by_source)), [f"Reference {i} (n={sizes[i]})" for i in by_source.index])
    ax.set_title("Predictor missingness by source study\nReporting patterns can confound row-random evaluation")
    fig.colorbar(im, ax=ax, label="Missing within study (%)", shrink=.7)
    fig.savefig(OUT / "tensile_missingness_by_reference.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [STUDENT_FILE, TENSILE_FILE]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    raw_s = pd.read_csv(STUDENT_FILE, sep=";", encoding="utf-8-sig")
    raw_s.columns = raw_s.columns.str.strip()
    raw_t = pd.read_excel(TENSILE_FILE, sheet_name="Tensile Data", header=1)
    raw_t.columns = raw_t.columns.str.strip()
    desc = pd.read_excel(TENSILE_FILE, sheet_name="Coulmn Descriptions", header=None)
    metadata = json.loads((ROOT / "data/reference/student_variables.json").read_text())
    s = raw_s.loc[raw_s.Target.isin(["Graduate", "Dropout"])].copy()
    t = raw_t.loc[raw_t[UTS].notna()].copy()
    sd, td = student_decisions(raw_s, s, metadata), tensile_decisions(raw_t, t, desc)
    save_table(sd, "student_column_decisions"); save_table(td, "tensile_column_decisions")
    issues, zero_rows = [], []

    def issue(dataset, row, column, value, rule, action, severity="review"):
        # CSV has one header row; Excel has two.
        issues.append({"dataset": dataset, "raw_index": int(row),
                       "source_row": int(row) + (2 if dataset == "student" else 3),
                       "column": column, "observed_value": str(value), "finding": rule,
                       "recommended_action": action, "severity": severity})

    for sem in (1, 2):
        prefix = f"Curricular units {sem}{'st' if sem == 1 else 'nd'} sem "
        enrolled, approved, grade, none = [prefix + f"({suffix})" for suffix in ["enrolled", "approved", "grade", "without evaluations"]]
        for label, frame in s.groupby("Target"):
            zero_rows.append({"semester": sem, "target": label, "class_rows": len(frame),
                              "zero_grade_count": int(frame[grade].eq(0).sum()),
                              "zero_grade_percent": frame[grade].eq(0).mean() * 100,
                              "zero_grade_vs_zero_approved_mismatches": int(frame[grade].eq(0).ne(frame[approved].eq(0)).sum())})
        for col in [approved, none]:
            for idx in s.index[s[col].gt(s[enrolled])]:
                issue("student", idx, col, s.loc[idx, col], f"Exceeds enrolled count ({s.loc[idx, enrolled]}); source-dependent consistency screen",
                      "Investigate source definition; preserve raw row until resolved.")
    # Check category codes against the UCI dictionary.
    definitions = {v["name"].strip().lower(): v for v in metadata["variables"]}
    for col in NOMINAL | BINARY:
        codes = {int(v) for v in re.findall(r"(?<!\d)(\d+)\s*[-–]\s*", definitions[col.lower()]["description"])}
        if not codes:
            raise ValueError(f"No documented codes parsed for {col}")
        for idx in s.index[~s[col].isin(codes)]:
            issue("student", idx, col, s.loc[idx, col], "Code absent from retrieved UCI dictionary", "Confirm source mapping; do not guess category.")
    for col, lo, hi in [("Application order", 0, 9), ("Previous qualification (grade)", 0, 200), ("Admission grade", 0, 200)]:
        for idx in s.index[~s[col].between(lo, hi)]:
            issue("student", idx, col, s.loc[idx, col], f"Outside documented range {lo}–{hi}", "Check source row.")

    ratio_rows = []
    for ratio, numerator in RATIOS.items():
        available = t[numerator].notna() & t["Width (mm)"].gt(0)
        expected = t[numerator] / t["Width (mm)"].where(t["Width (mm)"].gt(0))
        derivable = available & t[ratio].isna()
        # Screening tolerance; flagged rows need source review.
        conflicts = available & t[ratio].notna() & (t[ratio] - expected).abs().gt(.02)
        ratio_rows.append({"ratio": ratio, "observed_comparable": int((available & t[ratio].notna()).sum()),
                           "missing_derivable": int(derivable.sum()), "conflicts_at_absolute_0_02": int(conflicts.sum())})
        for idx in t.index[conflicts]:
            issue("tensile", idx, ratio, t.loc[idx, ratio], f"Observed ratio differs from dimensions: calculated {expected.loc[idx]:.6g}",
                  "Check reference article; neither ratio nor dimension is presumed correct.")
    tokens = []
    for col in td.loc[td.semantic_type.str.startswith("numeric"), "column"]:
        numeric = pd.to_numeric(t[col], errors="coerce")
        for token, count in t.loc[t[col].notna() & numeric.isna(), col].value_counts().items():
            tokens.append({"column": col, "token": token, "count": int(count),
                           "decision": "Preserve raw token; unit/range/limit/balance interpretation required before imputation."})
    geometry = list(RATIOS.values()) + ["Width (mm)", "Strain Rate (1/s)", "Grain size (um)"]
    for col in geometry:
        for idx in t.index[t[col].notna() & pd.to_numeric(t[col], errors="coerce").le(0)]:
            issue("tensile", idx, col, t.loc[idx, col], "Nonpositive measurement needs domain review", "Check source/geometry; do not replace automatically.")
    chemistry = [c for c in t if "(wt" in c]
    numeric_chem = t[chemistry].apply(pd.to_numeric, errors="coerce")
    # Sum known numeric components; blanks remain unknown.
    reported_sum = numeric_chem.sum(axis=1, min_count=1)
    for idx in t.index[reported_sum.gt(100.05)]:
        issue("tensile", idx, "reported numeric composition sum", reported_sum.loc[idx], "Known numeric components exceed 100.05 wt.%", "Check source composition and units; blanks remain unknown.")
    treatment = t[["Reference", "Treatment", "Treatment – Three Categories"]].dropna().copy()
    treatment["normalized_label"] = treatment.Treatment.str.strip().str.casefold().str.replace(r"\s+", " ", regex=True)
    ambiguous = treatment.groupby("normalized_label")["Treatment – Three Categories"].nunique()
    save_table(treatment.loc[treatment.normalized_label.isin(ambiguous.index[ambiguous.gt(1)])].drop_duplicates(), "treatment_mapping_review")
    save_table(pd.DataFrame(issues, columns=["dataset", "raw_index", "source_row", "column", "observed_value", "finding", "recommended_action", "severity"]), "quality_issues")
    save_table(pd.DataFrame(tokens), "numeric_token_review")
    save_table(pd.DataFrame(ratio_rows), "geometry_dependency_review")
    zero = pd.DataFrame(zero_rows); save_table(zero, "student_zero_grade_review")
    academic = [c for c in s if " sem " in c]
    ac, student_pairs = correlations(s, academic, "student_semester")
    numeric = ["Gauge Length (mm)", "Width (mm)", "Thickness (mm)", "Aspect Ratio (L/W)",
               "Aspect Ratio (T/W)", "Test Temperature (C)", "Strain Rate (1/s)", UTS]
    tc, tensile_pairs = correlations(t, numeric, "tensile_numeric")
    q1, q3 = t[UTS].quantile([.25, .75]); iqr = q3 - q1
    outliers = t[UTS].lt(q1 - 1.5 * iqr) | t[UTS].gt(q3 + 1.5 * iqr)
    flagged = t.loc[outliers, ["Reference", "Material", "Material Group", "Test Temperature (C)", UTS]].copy()
    flagged.insert(0, "source_row", flagged.index + 3); flagged.insert(0, "raw_index", flagged.index)
    save_table(flagged, "uts_outlier_review")
    numeric_profiles = []
    for dataset, frame, cols in [("student", s, [c for c in s if c not in NOMINAL | BINARY | {"Target"}]),
                                 ("tensile", t, td.loc[td.semantic_type.str.startswith("numeric"), "column"].tolist())]:
        for col in cols:
            v = pd.to_numeric(frame[col], errors="coerce")
            numeric_profiles.append({"dataset": dataset, "column": col,
                                     "unparsed_observed": int((frame[col].notna() & v.isna()).sum()),
                                     **v.describe().to_dict()})
    save_table(pd.DataFrame(numeric_profiles), "numeric_profiles")
    predictors = td.loc[td.role.eq("allowed predictor")]
    figures(s, t, ac, tc, predictors, zero)
    summary = {"student_raw_rows": len(raw_s), "student_binary_rows": len(s),
               "student_raw_columns": len(raw_s.columns), "student_missing_cells": int(raw_s.isna().sum().sum()),
               "student_duplicates": int(raw_s.duplicated().sum()), "student_class_counts": s.Target.value_counts().to_dict(),
               "student_unknown_counts": sd.loc[sd.coded_unknown_count.gt(0), ["column", "coded_unknown_count"]].to_dict("records"),
               "tensile_raw_rows": len(raw_t), "tensile_raw_columns": len(raw_t.columns), "uts_observed_rows": len(t),
               "uts_missing_rows": int(raw_t[UTS].isna().sum()), "tensile_duplicates": int(raw_t.duplicated().sum()),
               "uts_reference_count": int(t.Reference.nunique()), "uts_tukey_flags": int(outliers.sum()),
               "geometry": ratio_rows, "issue_records": len(issues),
               "student_high_correlation_pairs": student_pairs.head(8).to_dict("records"),
               "source_sha256": before, "transformations_applied": False, "models_trained": False,
               "exploratory_only": "Full-snapshot EDA informs hypotheses; no performance estimate or learned preprocessing is fitted."}
    assert set(sd.column) == set(raw_s.columns) and len(sd) == raw_s.shape[1]
    assert set(td.column) == set(raw_t.columns) and len(td) == raw_t.shape[1]
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert after == before, "Source data changed during the review"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source_sha256", "student_high_correlation_pairs"}}, indent=2))
    print(f"Saved column decisions, review tables and six figures to {OUT}")


if __name__ == "__main__":
    main()
