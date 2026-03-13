# COVID-19 Public Discussion Analysis

This project implements a full NLP pipeline on the Kaggle COVID-19 tweets dataset:

- Text preprocessing and normalization
- Feature representation with Bag-of-Words, TF-IDF, and Word2Vec
- Document similarity analysis using cosine similarity
- Evaluation and recommendation of representation methods

## 1) Setup

### Create environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Kaggle credentials

Use either kagglehub or kaggle CLI.
For kaggle CLI fallback, ensure your credentials are configured:

- Place kaggle.json in your user profile .kaggle folder.
- Example path on Windows: C:/Users/YourUser/.kaggle/kaggle.json

## 3) Run pipeline

```powershell
python run_pipeline.py
```

Optional smaller run:

```powershell
$env:PIPELINE_SAMPLE_SIZE="5000"
python run_pipeline.py
```

## 4) Expected outputs

- Cleaned dataset: data/processed/covid_tweets_cleaned.csv
- Similarity ranking: outputs/tweet_similarity_ranking.csv
- Method comparison: outputs/representation_comparison.csv
- Word2Vec model: outputs/word2vec_covid.model
- Text report: outputs/analysis_summary.txt

## 5) Notes for low-resource machines

- Keep sample size between 5,000 and 20,000 during development.
- Avoid converting sparse matrices to dense arrays.
- Reuse saved outputs instead of retraining every run.
