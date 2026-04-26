"""Compare the two candidate CSVs on the metrics that matter for claim-checking."""
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COVID_RE = re.compile(r"covid|sars[- ]?cov[- ]?2|coronavirus|ncov|2019[- ]?ncov", re.I)

FILES = [
    ("data.csv (2018-era)", os.path.join(HERE, "data.csv")),
    ("metadata.csv (2022)", os.path.join(HERE, "metadata.csv")),
]

def summarize(label, path):
    print(f"\n{'='*72}\n{label}  —  {os.path.getsize(path)/1e6:.1f} MB\n{'='*72}")
    # Peek at columns from header only
    head = pd.read_csv(path, nrows=0)
    print("Columns:", list(head.columns)[:20], "..." if len(head.columns) > 20 else "")
    print(f"Total columns: {len(head.columns)}")

    # Stream in chunks to count efficiently (especially for 1.65 GB file)
    rows = 0
    abstract_rows = 0
    covid_rows = 0
    covid_with_abstract = 0
    years = {}
    use_cols = [c for c in ["abstract", "title", "publish_time"] if c in head.columns]
    for chunk in pd.read_csv(path, usecols=use_cols, chunksize=50_000,
                             dtype=str, low_memory=False, on_bad_lines="skip"):
        rows += len(chunk)
        if "abstract" in chunk.columns:
            has_abs = chunk["abstract"].notna() & (chunk["abstract"].str.len() > 50)
            abstract_rows += int(has_abs.sum())
        text = chunk.get("abstract", pd.Series([""]*len(chunk))).fillna("") + " " + \
               chunk.get("title", pd.Series([""]*len(chunk))).fillna("")
        is_covid = text.str.contains(COVID_RE, na=False)
        covid_rows += int(is_covid.sum())
        covid_with_abstract += int((is_covid & has_abs).sum() if "abstract" in chunk.columns else 0)
        if "publish_time" in chunk.columns:
            yrs = pd.to_datetime(chunk.loc[is_covid, "publish_time"],
                                 errors="coerce").dt.year.dropna().astype(int)
            for y, c in yrs.value_counts().items():
                years[y] = years.get(y, 0) + int(c)

    print(f"Total rows                        : {rows:,}")
    print(f"Rows with usable abstract (>50 ch): {abstract_rows:,}  "
          f"({abstract_rows/rows*100:.1f}%)")
    print(f"COVID-related rows                : {covid_rows:,}  "
          f"({covid_rows/rows*100:.1f}%)")
    print(f"COVID rows WITH abstract          : {covid_with_abstract:,}  "
          f"<-- usable evidence pool")
    if years:
        print("Year distribution of COVID papers (top years):")
        for y in sorted(years, reverse=True)[:8]:
            print(f"  {y}: {years[y]:,}")

for label, path in FILES:
    summarize(label, path)
