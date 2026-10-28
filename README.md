# Week 4: Predictive Modeling Framework for Agricultural Yield Forecasting

A GitHub-ready starter project outlining and demonstrating a machine-learning workflow for forecasting crop yield.

## Project contents

- `reports/Week4_Predictive_Modeling_Framework.docx` — detailed assignment report.
- `src/train_model.py` — reusable training script that compares a baseline, linear regression, random forest, and gradient boosting when available.
- `data/sample_agricultural_yields.csv` — **synthetic demonstration data**. It is not real farm or government data.
- `requirements.txt` — Python dependencies.
- `.gitignore` — common generated files to exclude from Git.

## Research question

Can weather, soil, crop-management, and seasonal features help predict crop yield (tonnes per hectare) for a defined crop, location, and growing season?

## Quick start

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the example:

```bash
python src/train_model.py --data data/sample_agricultural_yields.csv --target yield_tonnes_per_hectare
```

The script reports test-set MAE, RMSE, and R² for each model. It uses a chronological split when a `season_year` column is available, keeping later seasons for testing. The sample dataset is synthetic and intended only to demonstrate code execution.

## Replace the sample data

For a real study, use a verified dataset with a clearly documented source and licence. Keep one row per defined crop/location/season observation, align measurement units, and avoid including variables that would not be available at the intended forecast date.

Recommended feature groups:
- Weather: rainfall, temperature, humidity, solar radiation, heat-stress days.
- Soil: pH, organic carbon, nutrient levels, texture, water-holding capacity.
- Management: crop variety, sowing date, irrigation, fertilizer application.
- Context: location, season, elevation, historical yield.
- Remote sensing: vegetation indices aligned to the forecast date.

## Methodological notes

- The target is yield per hectare, not total production.
- Preprocessing is fitted inside a scikit-learn pipeline to reduce leakage.
- The sample is chronological, but a real project should also consider grouped validation by farm/location and testing on unseen regions.
- Do not treat synthetic data metrics as evidence of real-world predictive performance.
- Choose the final model using validation results, domain requirements, reliability, and an untouched test set.

## Suggested next steps

1. Replace synthetic data with a documented real dataset.
2. Define the forecast date and prediction horizon.
3. Add time-aware or group-aware cross-validation.
4. Analyze residuals and performance by crop, region, and season.
5. Estimate prediction intervals and monitor performance after deployment.

## License

Add a licence if you intend to publish or reuse this project publicly. Check the licence of any real dataset separately.
