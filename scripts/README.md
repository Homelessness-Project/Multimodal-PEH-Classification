# ACLSRW25

This repository contains code and data for analyzing Reddit comments across multiple cities using LLMs (Llama 3.2 and Qwen 2.5) for classification and mitigation.

## Data Collection

### 1. Reddit Data Collection
Run the following script to collect Reddit data:
```bash
python scripts/get_reddit_data.py
```

**Note:** You'll need to:
- Replace `CLIENT_ID`, `CLIENT_SECRET`, and `USER_AGENT` with your Reddit API credentials
- Specify your target subreddit name
- The script outputs 3 CSVs in `data/<city>/reddit/`:
  - `all_comments.csv` (not included due to identifiable information)
  - `filtered_comments.csv` (not included due to identifiable information)
  - `statistics.csv` (included)

After data collection, run:
```bash
python scripts/random_reddit_sample.py
```
This generates a random set of 50 Reddit comments per city.

### 2. Deidentified Data
The deidentified dataset (500 comments total, 50 from each of 10 cities) is available at:
[`output/sampled_reddit_comments_by_city_deidentified.csv`](output/sampled_reddit_comments_by_city_deidentified.csv)

To generate this yourself:
```bash
python scripts/deidentify_comments.py
```

## Model Setup

### 3. Download Required Models
Download the following models from HuggingFace:
- [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## Annotation and Classification

### 4. Gold Standard / Soft Labeling
The annotation data is available in:
- Raw scores: [`annotation/raw_scores.csv`](annotation/raw_scores.csv)
- Processed outputs:
  - [`output/annotation/column_agreement_stats.csv`](output/annotation/column_agreement_stats.csv)
  - [`output/annotation/soft_labels.csv`](output/annotation/soft_labels.csv)

To generate these yourself:
```bash
python scripts/annotator_agreement.py
```

### 5. Classification
The classified comments are available in:
- [`output/classified_comments_llama.csv`](output/classified_comments_llama.csv)
- [`output/classified_comments_qwen.csv`](output/classified_comments_qwen.csv)

To run the classification yourself:
```bash
python scripts/llama_3_2_classify.py
python scripts/qwen_3_2_classify.py
```

### 6. Mitigation
The mitigated comments are available in:
- [`output/mitigated_comments_llama.csv`](output/mitigated_comments_llama.csv)
- [`output/mitigated_comments_qwen.csv`](output/mitigated_comments_qwen.csv)

To run the mitigation yourself:
```bash
python scripts/llama_3_2_mitigate.py
python scripts/qwen_3_2_mitigate.py
```

## Analysis

### 7. Statistics and Visualization
All statistics and charts are available in the [`output/charts/`](output/charts/) directory.

To generate these yourself:
```bash
python scripts/calculate_intercoder_reliability.py
```