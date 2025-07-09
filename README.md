# AAAI26

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

### 2. Gold Subset Deidentified Data (for LLM Evaluation)
The gold subset deidentified dataset (500 comments total, 50 from each of 10 cities) is available at:
[`output/gold_subset_reddit_comments_by_city_deidentified.csv`](output/gold_subset_reddit_comments_by_city_deidentified.csv)

**Note:** This file is used for LLM evaluation and is NOT the human-annotated gold standard. The true human gold standard is stored separately (see annotation section below).

To generate this yourself:
```bash
python scripts/deidentify_comments.py
```

## Model Setup

### 3. Download Required Models
Download the following models from HuggingFace:
- [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Gemma 3 4B It](https://huggingface.co/google/gemma-3-4b-it)
- [Phi-4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

## Annotation and Classification

### 4. Gold Standard / Soft Labeling
The human-annotated gold standard and soft label annotation data for Reddit are available in:
- Raw scores: [`annotation/reddit_raw_scores.csv`](annotation/reddit_raw_scores.csv)
- Processed outputs:
  - [`output/annotation/column_agreement_stats.csv`](output/annotation/column_agreement_stats.csv)
  - [`output/annotation/reddit_soft_labels.csv`](output/annotation/reddit_soft_labels.csv)

Other sources (e.g., X, news_articles, meeting_minutes) may have their own raw scores and soft label files in the future.

To generate these yourself:
```bash
python scripts/annotator_agreement.py
```

## Classification and Mitigation

**Note:** The only valid options for `--source` are: `reddit`, `x`, `news_articles`, and `meeting_minutes`. The only valid options for `--model` are: `llama`, `qwen`, `gemma3`, and `phi4`.

### 5. Classification
You can classify comments using any of the four supported models with a single script. You must specify the data source and dataset:

**Zero-shot (default):**
```bash
python scripts/classify_comments.py --model llama --source reddit --dataset gold_subset
python scripts/classify_comments.py --model qwen --source reddit --dataset gold_subset
python scripts/classify_comments.py --model gemma3 --source reddit --dataset gold_subset
python scripts/classify_comments.py --model phi4 --source reddit --dataset gold_subset
```

**Few-shot (with examples):**
```bash
python scripts/classify_comments.py --model llama --source reddit --dataset gold_subset --few_shot reddit
python scripts/classify_comments.py --model qwen --source reddit --dataset gold_subset --few_shot reddit
python scripts/classify_comments.py --model gemma3 --source reddit --dataset gold_subset --few_shot reddit
python scripts/classify_comments.py --model phi4 --source reddit --dataset gold_subset --few_shot reddit
```
- The `--model` argument specifies which model to use (`llama`, `qwen`, `gemma3`, or `phi4`).
- The `--source` argument specifies the data source (`reddit`, `x`, `news_articles`, `meeting_minutes`).
- The `--dataset` argument specifies which dataset to use (`all`, `gold_subset`).
- For Reddit, `--dataset gold_subset` uses the gold subset file (`output/gold_subset_reddit_comments_by_city_deidentified.csv`) for LLM evaluation. This is NOT the human gold standard.
- The `--few_shot` argument appends five few-shot examples to the end of the prompt. Supported values: `reddit`, `x`, `news_articles`, `meeting_minutes` (when defined in `utils.py`).
- If `--few_shot` is not specified, zero-shot classification is used.
- The output will be saved to `output/classified_comments_{source}_{dataset}_{model}.csv` by default, or `output/classified_comments_{source}_{dataset}_{model}_fewshot.csv` if few-shot is used (e.g., `output/classified_comments_reddit_gold_subset_llama.csv`).
- You can override the input or output file with `--input` and `--output` arguments.

### 6. Mitigation
To mitigate and reclassify comments using any of the four models (mitigation always includes reclassification):
```bash
python scripts/mitigate_comments.py --model llama --source reddit --dataset gold_subset
python scripts/mitigate_comments.py --model qwen --source reddit --dataset gold_subset
python scripts/mitigate_comments.py --model gemma3 --source reddit --dataset gold_subset
python scripts/mitigate_comments.py --model phi4 --source reddit --dataset gold_subset
```
- The `--model` argument specifies which model to use (`llama`, `qwen`, `gemma3`, or `phi4`).
- For Reddit, `--dataset gold_subset` uses the gold subset file (`output/gold_subset_reddit_comments_by_city_deidentified.csv`) for LLM evaluation. This is NOT the human gold standard.
- The output will be saved to `output/mitigated_comments_{source}_{dataset}_{model}.csv` by default, or `output/mitigated_comments_{source}_{dataset}_{model}_fewshot.csv` if few-shot is used (e.g., `output/mitigated_comments_reddit_gold_subset_llama.csv`).
- Mitigation always includes reclassification of the mitigated comments.

## Analysis

### 7. Statistics and Visualization
All statistics and charts are available in the [`output/charts/`](output/charts/) directory.

To generate these yourself:
```bash
python scripts/calculate_intercoder_reliability.py
```