# Data-Science-Team-Project-Midterm
Team Members -
Mallory Roussel,
Noah Torres,
Greyson Alegria,
Shane Roth

## Week 3 data review

The four-person team roster above was confirmed on October 16, 2026. All four
members remain on the project. The supplied rubric describes three-person teams;
model ownership will be planned across the confirmed four-person team.

The October 16–18 review is documented in [docs/week3_review.md](docs/week3_review.md)
and [week3_data_review.ipynb](week3_data_review.ipynb). All 37 student columns and
55 tensile columns have review decisions in `outputs/week3/`. That folder also
contains six EDA figures, source-row issue registers, correlation sample counts,
and source-data checksums. Decisions remain proposals for the next preprocessing
stage; raw values are unchanged and no models have been trained.

Reproduce the review from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-week3.txt
.venv/bin/python scripting/week3_eda.py
```

The script resolves data paths relative to itself and can also run from another
working directory. It reads local source files and writes only to `outputs/week3/`.
The original scripts require their own dependencies, including seaborn.

The local data was copied from the synced project snapshot ending at `45afb22`.
The synced source was left unchanged. This review does not verify GitHub repository
privacy or remote publication. Remote publication is separate from this local
working copy.
