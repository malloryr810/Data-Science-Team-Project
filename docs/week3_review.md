# Week 3: data interpretation and focused EDA

Review date: October 16, 2026. Scope: the interpretation and EDA work planned for October 16–18. This is a technical review record, not the final three-page submission. No missing values have been filled, source measurements corrected, observations removed as outliers, or models trained in this stage.

## Research questions and populations

Classification: how well can permitted demographic, socioeconomic, enrollment and academic variables distinguish students recorded as Graduate or Dropout? Remove Enrolled as required by the assignment. The resulting population excludes students with an unresolved final outcome; findings do not automatically generalize to every currently enrolled student.

Regression: how well can pre-test material, processing, irradiation, geometry and test-condition information predict observed UTS in MPa? Exclude records without an observed UTS target. This creates an observed-target subset that may differ from the full collection; missing target values must not be imputed to create labels.

The supplied rubric governs permitted features. It allows both semesters of academic variables. Use them in the main retrospective classification analysis, subject to evidence-based selection. An enrollment-only comparison is optional. The classification task is predictive; associations with scholarships, fees or grades are not causal effects.

## Existing work

The starting snapshot contains binary student filtering, null/duplicate checks, class balance, academic correlations, tensile checks, forbidden-outcome exclusions, predictor missingness and UTS outlier flags. Local source commits include `5ef28f0` (semester correlations), `50984d3` (regression missingness) and `45afb22` (UTS outliers). This is evidence of local history, not verification that every change has been pushed or that three weeks of individual collaboration have occurred.

The original synced project remains unchanged. Source data and existing scripts were copied into the DSFIP working repository. The review is reproducible using `scripting/week3_eda.py`; `week3_data_review.ipynb` presents the outputs. The initial scripts retain their original behavior; their comments have been simplified.

## Column-level missing-value decisions

`outputs/week3/student_column_decisions.csv` covers all 37 student columns. `tensile_column_decisions.csv` covers all 55 workbook columns. Each row records meaning, semantic type, role, availability, observed missingness, zero counts, dependencies and a proposed treatment. Statistics in these tables describe the binary student or observed-UTS population; raw missing counts are separately labeled. Proposed actions are not completed transformations.

Student rules:

- Numeric-looking course, application, qualification, occupation and nationality codes represent categories. They must not be median-filled or interpreted as ordered magnitudes merely because their storage type is integer.
- Parental qualification `34` is documented Unknown. Parental occupation `99` is documented blank. Preserve their missing-information meaning as explicit unknown categories. Occupation `0` means Student. Application order `0` means first choice. Valid binary zeros mean the documented negative/category state.
- Zero semester grades coincide with zero approved units in this snapshot. This supports a structural-recording hypothesis, not an established definition of why the grade is zero. Keep the original zero; investigate a no-approved-units indicator and confirm source semantics. Do not silently substitute a mean grade.
- Negative inflation and GDP observations are not automatically erroneous or missing. Enrolled is an excluded target category, not a missing predictor or a negative class.
- The UCI dictionary has apparent inconsistencies: grades are typed Integer despite fractional observed values, and the second-semester without-evaluations description mentions the first semester. Use observed numeric grades and explicit semester column names, retaining these documentation caveats.

Tensile rules:

- Composition blanks do not establish zero concentration. The local file contains numerical values, ranges, limits and balance markers. Parse all representations before choosing an imputation method.
- One wt.% equals 10,000 ppm. Thus `<40 ppm` is an upper bound of 0.004 wt.%, and `150-400ppm` spans 0.015–0.040 wt.%. An upper bound is not an exact concentration. A range midpoint is an approximation that must be declared and tested, not an observed measurement. Preserve the raw token and uncertainty flags/bounds.
- `bal` means a balance component. Calculate `100 - sum(other components)` only when those components are sufficiently known; missing components cannot be silently assumed zero. The published paper describes normalized compositions, but the local workbook still has text tokens. Keep this version discrepancy explicit.
- Missing manufacturing, treatment, cooling, grain and irradiation information can reflect omissions in source articles. Unreported, confirmed absent and not applicable are different states. A reporting indicator must not be named an irradiation/treatment yes/no indicator.
- Zero irradiation dose is an observation; a missing dose is unknown. Zero or negative Celsius temperatures can be valid. Do not fill an unknown test temperature with zero or presumed room temperature.
- If ordinary numeric features are retained, a training-fitted median plus missingness indicator is a candidate baseline. It is not a universal physical interpretation. Very sparse fields need an explicit coverage/relevance decision rather than a blind percentage cutoff. Unknown categories can be encoded explicitly.

## Leakage and evaluation decisions

| Risk | Evidence/limit | Required decision |
|---|---|---|
| Student final outcome | Target and target-derived variables reveal the answer | Exclude from every input matrix and derived feature |
| Semester timing | Semester records cannot be available at enrollment; individual dropout dates are absent | Describe the main model as using recorded academic information; do not claim verified prospective early-warning performance |
| Financial/status timing | Snapshot dates for debt, tuition and scholarship status are absent | Permitted by the rubric; retain for investigation and document timing uncertainty |
| Preprocessing leakage | Imputation, scaling, encoding, resampling and selection can learn from test records | Fit inside training folds; keep the held-out test set out of model choices |
| Tensile outcomes | Yield strength and elongation outcomes come from the same experiment | Exclude UTS as input, Yield Strength, Uniform Elongation and Total Elongation |
| Reference/study effects | Source papers contain repeated conditions and distinct reporting patterns | Exclude Reference as predictor; evaluate disjoint study groups when claiming generalization to new studies |
| Full-snapshot EDA | This descriptive review sees the supplied snapshot | Treat it as hypothesis generation, not independent performance evidence; fit learned choices on training data only |

For students, plan a fixed stratified holdout with stratified cross-validation within training data. Set Dropout as the positive class. Report precision, recall, F1, confusion matrix and supporting accuracy; compare with a dummy baseline. The moderate class imbalance does not justify automatic oversampling. Compare class weighting on training folds if needed. Any resampling stays inside folds.

For tensile prediction, plan a fixed study-group holdout and grouped training folds if unseen-study generalization is the goal. Check sample size, material coverage and unfamiliar categories in each split. Reference grouping tests a different question from random-row interpolation; state the question and tradeoff. Report MAE and RMSE in MPa and R-squared. Do not compare model scores from different holdouts as if they were equivalent.

## Relationships and unusual observations

The reviewed student population contains 3,630 records (2,209 Graduate; 1,421 Dropout), with no literal null cells or exact duplicate rows. Coded unknown information remains: 127 mother-qualification, 109 father-qualification, 15 mother-occupation and 17 father-occupation records. Corresponding semester credited, enrolled and approved counts have Pearson correlations of approximately 0.947, 0.941 and 0.916 respectively. These support a redundancy investigation, not automatic removal.

The tensile workbook contains 1,050 records, 489 with UTS and 561 without. There are no exact duplicate rows. The observed-UTS subset spans 22 references. Missing ratios potentially recoverable from recorded dimensions number 8 (L/W), 17 (R/W) and 19 (T/W). Their source dimensions must still be validated.

The issue register has ten entries across eight distinct source rows: three student count discrepancies; three geometry conflicts in one tensile row (Excel row 58); and four tensile rows with numeric reported composition totals of 100.74 wt.% (Excel rows 16–19). Unknown components cannot reduce an already excessive known-component sum, so those four records need source review. The screening tolerance is 100.05 wt.%, not a certified measurement tolerance. Seventeen UTS observations fall outside pooled Tukey fences and remain retained.

Academic correlations are descriptive and do not establish leakage. The full matrix, pair sample counts and sorted pairs are saved. Compare retaining both semesters against justified reduced or derived representations using training folds. Avoid automatic deletion at an arbitrary correlation threshold. Do not calculate Pearson relationships between nominal category codes as if code distances were meaningful.

Grade, approved-unit, enrollment and evaluation features partly describe the same academic process. A pass-rate feature would need a nonzero enrollment denominator; zero enrollment has a distinct meaning and must not produce an infinite value. Evaluation counts can include multiple assessments per unit, so evaluations exceeding enrolled units is not itself a proven error.

Geometry ratios depend on observed dimensions. Missing ratios can sometimes be calculated deterministically. The review flags differences above an exploratory absolute tolerance of 0.02; this is a screening tolerance, not a source-certified error bound. One row disagrees with all three ratios. Neither its ratios nor dimensions are presumed correct. In a future pipeline, choose dimensions or a deliberate ratio representation rather than filling both independently and creating inconsistent geometry.

Material/group and detailed/grouped treatment information overlap. Inspect mapping consistency before dropping either. The treatment mapping review lists normalized labels associated with different groups; generic descriptions such as Annealed may lack enough detail to resolve them. Do not merge treatment categories using text matching alone.

UTS Tukey flags identify possible extremes, not invalid tests. Compare extremes by material group and test conditions. Retain them pending source evidence. The material-group plot shows why a single pooled outlier cutoff is an incomplete basis for deletion. Correlations use pairwise observed data; their sample counts differ and missingness may bias apparent relationships.

## Source questions and next actions

The reproducible `quality_issues.csv` gives dataset, zero-based raw index, physical CSV/Excel row, observed value and proposed investigation. `numeric_token_review.csv` lists every nonnumeric token in otherwise numeric fields. `geometry_dependency_review.csv` records derivable counts and conflicts. No issue has been silently corrected.

1. Inspect the flagged tensile geometry row against Reference 3. Resolve the original measurements before deriving or replacing ratios.
2. Confirm the definition and recording time of zero grades and the student without-evaluation count discrepancies. Preserve records while unresolved.
3. Confirm unreported versus not-applicable treatment/irradiation values from source papers where feasible. Otherwise retain Unknown or omit unsupported features with justification.
4. Confirm the local workbook version and decide explicit treatment of ranges, upper limits and balance values. Audit every resulting numerical conversion before imputation.
5. The user confirmed the four-person roster on October 16: Mallory Roussel, Noah Torres, Greyson Alegria and Shane Roth. All four remain on the project. The supplied rubric describes three-person teams; individual responsibilities must reflect the confirmed four-person group. Actual contributions have not been inferred.
6. Implement preprocessing on October 19–20, followed by initial model training on October 21 and midterm report/slides on October 22. Proposed model pairs remain logistic regression/Ridge, random forest classifier/regressor, and SVC/SVR. Assign named model and shared-pipeline responsibilities across all four members before training; the model list and ownership remain provisional. Tuning and final held-out comparisons belong to the later modeling stage.

Weekly progress reports should describe actual changes, technical challenges, implemented solutions versus proposed ones, next tasks and named individual responsibilities. The midterm three-page report and slides are separate deliverables. Record contributions from actual Git history; local commits and remote pushes are separate actions.

## Sources

- Supplied course rubric and instructor feedback, received October 16, 2026.
- Realinho et al., UCI student dataset and variable dictionary, CC BY 4.0: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>. Retrieved definitions are saved in `data/reference/student_variables.json` for offline reuse.
- Romero and Liao (2025), student analysis: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325047>.
- Li et al. (2025), tensile dataset description: <https://www.nature.com/articles/s41597-024-04329-2.pdf>. The local workbook also includes References and Coulmn Descriptions sheets.
- Local source data, retained unchanged; SHA-256 hashes are recorded in `outputs/week3/summary.json`.
