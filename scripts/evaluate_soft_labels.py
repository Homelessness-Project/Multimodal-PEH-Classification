import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Paths
soft_gold_path = "output/annotation/reddit_soft_labels.csv"
llama_zero_path = "output/reddit/llama/classified_comments_reddit_gold_subset_llama_none_flags.csv"
llama_few_path = "output/reddit/llama/classified_comments_reddit_gold_subset_llama_reddit_flags.csv"
qwen_zero_path = "output/reddit/qwen/classified_comments_reddit_gold_subset_qwen_none_flags.csv"
qwen_few_path = "output/reddit/qwen/classified_comments_reddit_gold_subset_qwen_reddit_flags.csv"

# Load soft gold and binarize at 1.0 (positive only if value is 1)
soft_gold = pd.read_csv(soft_gold_path)
soft_gold_bin = soft_gold.copy()
for col in soft_gold.columns:
    if col not in ['Comment', 'City']:
        soft_gold_bin[col] = (soft_gold[col] == 1.0).astype(int)

# Map gold columns to match model output columns
col_map = {}
for col in soft_gold_bin.columns:
    if col in ['Comment', 'City']:
        col_map[col] = col
    elif col == 'racist':
        col_map[col] = 'Racist_Flag'
    elif col in ['money aid allocation', 'government critique', 'societal critique']:
        col_map[col] = f'Critique_{col}'
    elif col in ['solutions/interventions']:
        col_map[col] = f'Response_{col}'
    elif col in ['personal interaction', 'media portrayal', 'not in my backyard', 'harmful generalization', 'deserving/undeserving']:
        col_map[col] = f'Perception_{col}'
    else:
        col_map[col] = f'Comment_{col}'
soft_gold_bin = soft_gold_bin.rename(columns=col_map)

# Load predictions
def load_pred(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None

llama_zero = load_pred(llama_zero_path)
llama_few = load_pred(llama_few_path)
qwen_zero = load_pred(qwen_zero_path)
qwen_few = load_pred(qwen_few_path)

# Find flag columns (must be present in both gold and pred)
def get_flag_columns(df):
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in [
        'Comment_', 'Critique_', 'Response_', 'Perception_', 'Racist_Flag'
    ])]

# Use only columns present in all predictions and gold
all_preds = [df for df in [llama_zero, llama_few, qwen_zero, qwen_few] if df is not None]
flag_cols = set(get_flag_columns(soft_gold_bin))
for pred in all_preds:
    flag_cols = flag_cols & set(get_flag_columns(pred))
flag_cols = sorted(flag_cols)

# Evaluation function
def evaluate(gold, pred, flag_cols, prefix):
    results = {}
    for col in flag_cols:
        y_true = gold[col].values
        y_pred = pred[col].values if col in pred else None
        if y_pred is not None and len(y_true) == len(y_pred):
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            results[col] = {
                f'{prefix}_accuracy': acc,
                f'{prefix}_precision': prec,
                f'{prefix}_recall': rec,
                f'{prefix}_f1': f1
            }
    return pd.DataFrame(results).T

# Evaluate all
dfs = []
if llama_zero is not None:
    dfs.append(evaluate(soft_gold_bin, llama_zero, flag_cols, 'llama_zero'))
if llama_few is not None:
    dfs.append(evaluate(soft_gold_bin, llama_few, flag_cols, 'llama_few'))
if qwen_zero is not None:
    dfs.append(evaluate(soft_gold_bin, qwen_zero, flag_cols, 'qwen_zero'))
if qwen_few is not None:
    dfs.append(evaluate(soft_gold_bin, qwen_few, flag_cols, 'qwen_few'))

# Merge all results on index (label)
from functools import reduce
if dfs:
    result_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
    result_df.index.name = 'label'
    out_path = "output/annotation/eval_soft_all_models.csv"
    result_df.to_csv(out_path)
    print(f"Saved all results to {out_path}")

    # Save just F1s
    f1_cols = [col for col in result_df.columns if col.endswith('_f1')]
    f1_df = result_df[f1_cols].copy()
    f1_df.to_csv("output/annotation/eval_soft_all_models_f1.csv")
    print("Saved F1-only results to output/annotation/eval_soft_all_models_f1.csv")
else:
    print("No results to save!") 