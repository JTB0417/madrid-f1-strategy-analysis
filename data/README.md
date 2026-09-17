# Data files

The repository includes the processed CSV results used for review. Raw OpenF1 API downloads are intentionally excluded to keep the repository lightweight and avoid republishing a large third-party dataset.

Generate the raw files locally before running the analysis:

```bash
python fetch_data.py
python analyze_race.py
```

The download script retrieves the Madrid race session with OpenF1 session key `11369`.
