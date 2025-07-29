import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
import argparse
import os

# Create output directories if they don't exist
os.makedirs('output/charts', exist_ok=True)

# Mapping from soft label columns to classification columns
field_map = {
    'ask a genuine question': 'Comment_ask a genuine question',
    'ask a rhetorical question': 'Comment_ask a rhetorical question',
    'provide a fact or claim': 'Comment_provide a fact or claim',
    'provide an observation': 'Comment_provide an observation',
    'express their opinion': 'Comment_express their opinion',
    'express others opinions': 'Comment_express others opinions',
    'money aid allocation': 'Critique_money aid allocation',
    'government critique': 'Critique_government critique',
    'societal critique': 'Critique_societal critique',
    'solutions/interventions': 'Response_solutions/interventions',
    'personal interaction': 'Perception_personal interaction',
    'media portrayal': 'Perception_media portrayal',
    'not in my backyard': 'Perception_not in my backyard',
    'harmful generalization': 'Perception_harmful generalization',
    'deserving/undeserving': 'Perception_deserving/undeserving',
    'racist': 'racist'
}

# File paths
SOFT_LABELS_FILE = 'output/annotation/reddit_soft_labels.csv'

# Model file paths - zero-shot (none) and few-shot (reddit)
MODEL_FILES = {
    'Llama': {
        'zero_shot': 'output/reddit/llama/classified_comments_reddit_gold_subset_llama_none_flags.csv',
        'few_shot': 'output/reddit/llama/classified_comments_reddit_gold_subset_llama_reddit_flags.csv'
    },
    'Qwen': {
        'zero_shot': 'output/classified_comments_reddit_gold_subset_qwen.csv',
        'few_shot': 'output/reddit/qwen/classified_comments_reddit_gold_subset_qwen_reddit_flags.csv'
    },
    'GPT-4': {
        'zero_shot': 'output/reddit/gpt4/classified_comments_reddit_gold_subset_gpt4_none_flags.csv',
        'few_shot': 'output/reddit/gpt4/classified_comments_reddit_gold_subset_gpt4_reddit_flags.csv'
    },
    'Grok': {
        'zero_shot': 'output/reddit/grok/classified_comments_reddit_gold_subset_grok_none_flags.csv',
        'few_shot': 'output/reddit/grok/classified_comments_reddit_gold_subset_grok_reddit_flags.csv'  # Will be created when few-shot is run
    },
    'Phi-4': {
        'zero_shot': 'output/classified_comments_reddit_gold_subset_phi4.csv',
        'few_shot': 'output/reddit/phi4/classified_comments_reddit_gold_subset_phi4_reddit_flags.csv'
    }
}

def load_model_data(model_name, shot_type):
    """Load model data for a specific model and shot type."""
    file_path = MODEL_FILES[model_name][shot_type]
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {model_name} {shot_type}: {len(df)} comments")
        return df
    except FileNotFoundError:
        print(f"File not found for {model_name} {shot_type}: {file_path}")
        return None

def calculate_metrics_for_model(model_df, soft_labels_df, field):
    """Calculate comprehensive metrics for a specific model and field."""
    if model_df is None:
        return {'f1': np.nan, 'precision': np.nan, 'recall': np.nan, 'balanced_f1': np.nan, 'pred_positives': np.nan, 'true_positives': np.nan, 'actual_positives': np.nan}
    
    try:
        # Get soft labels for this field
        soft_values = pd.to_numeric(soft_labels_df[field], errors='coerce').fillna(0)
        soft_bin = (soft_values == 1).astype(int)  # 1 if soft label is 1, else 0
        
        # Get model predictions for this field
        if field == 'racist':
            # Handle racist field specially
            if 'Racist_Flag' in model_df.columns:
                model_pred = model_df['Racist_Flag'].fillna(0).astype(int)
            else:
                # Extract from racist text field
                def extract_racist_value(text):
                    if pd.isna(text):
                        return 0
                    text = str(text).lower().strip()
                    if text == "yes" or "racist: yes" in text:
                        return 1
                    elif text == "no" or "racist: no" in text:
                        return 0
                    return 0
                model_pred = model_df['racist'].apply(extract_racist_value)
        else:
            # Handle other fields
            field_col = field_map[field]
            if field_col in model_df.columns:
                model_pred = pd.to_numeric(model_df[field_col], errors='coerce').fillna(0).astype(int)
            else:
                model_pred = np.zeros(len(model_df))
        
        # Calculate metrics
        if len(model_pred) == len(soft_bin):
            f1 = f1_score(soft_bin, model_pred, zero_division=0)
            precision = precision_score(soft_bin, model_pred, zero_division=0)
            recall = recall_score(soft_bin, model_pred, zero_division=0)
            
            # Calculate balanced F1 (macro F1) - handles class imbalance better
            balanced_f1 = f1_score(soft_bin, model_pred, average='macro', zero_division=0)
            
            pred_positives = np.sum(model_pred)
            true_positives = np.sum((soft_bin == 1) & (model_pred == 1))
            
            return {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'balanced_f1': balanced_f1,
                'pred_positives': pred_positives,
                'true_positives': true_positives,
                'actual_positives': np.sum(soft_bin)
            }
        else:
            return {'f1': np.nan, 'precision': np.nan, 'recall': np.nan, 'balanced_f1': np.nan, 'pred_positives': np.nan, 'true_positives': np.nan, 'actual_positives': np.nan}
    except Exception as e:
        print(f"Error calculating metrics for {field}: {e}")
        return {'f1': np.nan, 'precision': np.nan, 'recall': np.nan, 'balanced_f1': np.nan, 'pred_positives': np.nan, 'true_positives': np.nan, 'actual_positives': np.nan}

def calculate_f1_for_model(model_df, soft_labels_df, field):
    """Calculate F1 score for a specific model and field."""
    metrics = calculate_metrics_for_model(model_df, soft_labels_df, field)
    return metrics['f1']

def main():
    parser = argparse.ArgumentParser(description='Calculate F1 scores for multiple models with zero-shot and few-shot')
    parser.add_argument('--shot_type', choices=['zero_shot', 'few_shot', 'both'], default='both', 
                       help='Which shot type to analyze (default: both)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed precision/recall breakdown')
    parser.add_argument('--balanced_only', action='store_true', help='Output only balanced F1 scores')
    args = parser.parse_args()
    
    # Load soft labels
    print("Loading soft labels...")
    soft_labels_df = pd.read_csv(SOFT_LABELS_FILE)
    print(f"Loaded {len(soft_labels_df)} soft label samples")
    
    # Show class distribution for problematic fields
    print("\n=== Class Distribution for Problematic Fields ===")
    for field in ['deserving/undeserving', 'racist']:
        values = soft_labels_df[field].value_counts()
        print(f"{field}: {values.to_dict()}")
    
    # Fields to analyze
    fields = list(field_map.keys())
    
    # Determine which shot types to analyze
    shot_types = ['zero_shot', 'few_shot'] if args.shot_type == 'both' else [args.shot_type]
    
    # Calculate metrics for all models and shot types
    results = []
    detailed_results = []
    balanced_results = []
    
    for shot_type in shot_types:
        print(f"\n=== Analyzing {shot_type} results ===")
        
        for model_name in MODEL_FILES.keys():
            print(f"\nProcessing {model_name}...")
            
            # Load model data
            model_df = load_model_data(model_name, shot_type)
            
            # Calculate metrics for each field
            for field in fields:
                metrics = calculate_metrics_for_model(model_df, soft_labels_df, field)
                
                results.append({
                    'Model': model_name,
                    'Shot_Type': shot_type,
                    'Field': field,
                    'F1_Score': metrics['f1'],
                    'Balanced_F1': metrics['balanced_f1']
                })
                
                # Also create balanced-only results if requested
                if args.balanced_only:
                    balanced_results.append({
                        'Model': model_name,
                        'Shot_Type': shot_type,
                        'Field': field,
                        'Balanced_F1': metrics['balanced_f1']
                    })
                
                detailed_results.append({
                    'Model': model_name,
                    'Shot_Type': shot_type,
                    'Field': field,
                    'F1_Score': metrics['f1'],
                    'Balanced_F1': metrics['balanced_f1'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'Pred_Positives': metrics['pred_positives'],
                    'True_Positives': metrics['true_positives'],
                    'Actual_Positives': metrics['actual_positives']
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    detailed_df = pd.DataFrame(detailed_results)
    
    # Save detailed results
    results_df.to_csv('output/charts/multi_model_f1_detailed.csv', index=False)
    detailed_df.to_csv('output/charts/multi_model_f1_detailed_metrics.csv', index=False)
    print(f"\nDetailed results saved to output/charts/multi_model_f1_detailed.csv")
    print(f"Detailed metrics saved to output/charts/multi_model_f1_detailed_metrics.csv")
    
    # Save balanced-only results if requested
    if args.balanced_only:
        balanced_df = pd.DataFrame(balanced_results)
        balanced_df.to_csv('output/charts/multi_model_balanced_f1_only.csv', index=False)
        print(f"Balanced F1 only results saved to output/charts/multi_model_balanced_f1_only.csv")
    
    # Create summary pivot table
    if args.balanced_only:
        # Use balanced F1 for summary
        value_col = 'Balanced_F1'
        output_file = 'output/charts/multi_model_balanced_f1_summary.csv'
    else:
        # Use regular F1 for summary
        value_col = 'F1_Score'
        output_file = 'output/charts/multi_model_f1_summary.csv'
    
    if args.shot_type == 'both':
        summary_df = results_df.pivot_table(
            index='Field', 
            columns=['Model', 'Shot_Type'], 
            values=value_col, 
            aggfunc='first'
        )
    else:
        summary_df = results_df.pivot_table(
            index='Field', 
            columns='Model', 
            values=value_col, 
            aggfunc='first'
        )
    
    summary_df.to_csv(output_file)
    print(f"Summary results saved to {output_file}")
    
    # Print summary
    if args.balanced_only:
        print("\n=== Balanced F1 Scores Summary ===")
    else:
        print("\n=== F1 Scores Summary ===")
    if args.shot_type == 'both':
        # Print zero-shot results
        zero_shot_results = results_df[results_df['Shot_Type'] == 'zero_shot']
        print("\nZero-shot Results:")
        for field in fields:
            field_results = zero_shot_results[zero_shot_results['Field'] == field]
            print(f"{field:<25}", end="")
            for model_name in MODEL_FILES.keys():
                model_result = field_results[field_results['Model'] == model_name]
                if not model_result.empty:
                    if args.balanced_only:
                        f1 = model_result['Balanced_F1'].iloc[0]
                    else:
                        f1 = model_result['F1_Score'].iloc[0]
                    print(f"{f1:<8.3f}", end="")
                else:
                    print(f"{'N/A':<8}", end="")
            print()
        
        # Print few-shot results
        few_shot_results = results_df[results_df['Shot_Type'] == 'few_shot']
        print("\nFew-shot Results:")
        for field in fields:
            field_results = few_shot_results[few_shot_results['Field'] == field]
            print(f"{field:<25}", end="")
            for model_name in MODEL_FILES.keys():
                model_result = field_results[field_results['Model'] == model_name]
                if not model_result.empty:
                    if args.balanced_only:
                        f1 = model_result['Balanced_F1'].iloc[0]
                    else:
                        f1 = model_result['F1_Score'].iloc[0]
                    print(f"{f1:<8.3f}", end="")
                else:
                    print(f"{'N/A':<8}", end="")
            print()
    else:
        # Print single shot type results
        shot_results = results_df[results_df['Shot_Type'] == args.shot_type]
        print(f"\n{args.shot_type.replace('_', ' ').title()} Results:")
        for field in fields:
            field_results = shot_results[shot_results['Field'] == field]
            print(f"{field:<25}", end="")
            for model_name in MODEL_FILES.keys():
                model_result = field_results[field_results['Model'] == model_name]
                if not model_result.empty:
                    if args.balanced_only:
                        f1 = model_result['Balanced_F1'].iloc[0]
                    else:
                        f1 = model_result['F1_Score'].iloc[0]
                    print(f"{f1:<8.3f}", end="")
                else:
                    print(f"{'N/A':<8}", end="")
            print()
    
    # Show detailed breakdown for problematic fields
    if args.detailed:
        print("\n=== Detailed Breakdown for Problematic Fields ===")
        for field in ['deserving/undeserving', 'racist']:
            print(f"\n{field.upper()}:")
            field_detailed = detailed_df[detailed_df['Field'] == field]
            for _, row in field_detailed.iterrows():
                print(f"  {row['Model']} ({row['Shot_Type']}): F1={row['F1_Score']:.3f}, Balanced_F1={row['Balanced_F1']:.3f}, "
                      f"Precision={row['Precision']:.3f}, Recall={row['Recall']:.3f}, "
                      f"Pred={row['Pred_Positives']}, True={row['True_Positives']}, Actual={row['Actual_Positives']}")
    
    # Calculate mean F1 scores per model and shot type
    if args.balanced_only:
        print("\n=== Mean Balanced F1 Scores by Model and Shot Type ===")
        mean_results = results_df.groupby(['Model', 'Shot_Type'])['Balanced_F1'].mean().reset_index()
        mean_results = mean_results.pivot(index='Model', columns='Shot_Type', values='Balanced_F1')
        mean_results.to_csv('output/charts/multi_model_balanced_f1_means.csv')
        print(mean_results)
        print("\nMean Balanced F1 scores saved to output/charts/multi_model_balanced_f1_means.csv")
    else:
        print("\n=== Mean F1 Scores by Model and Shot Type ===")
        mean_results = results_df.groupby(['Model', 'Shot_Type'])['F1_Score'].mean().reset_index()
        mean_results = mean_results.pivot(index='Model', columns='Shot_Type', values='F1_Score')
        mean_results.to_csv('output/charts/multi_model_f1_means.csv')
        print(mean_results)
        print("\nMean F1 scores saved to output/charts/multi_model_f1_means.csv")
    
        # Calculate mean Balanced F1 scores per model and shot type (only if not balanced_only)
        if not args.balanced_only:
            print("\n=== Mean Balanced F1 Scores by Model and Shot Type ===")
            balanced_mean_results = results_df.groupby(['Model', 'Shot_Type'])['Balanced_F1'].mean().reset_index()
            balanced_mean_results = balanced_mean_results.pivot(index='Model', columns='Shot_Type', values='Balanced_F1')
            balanced_mean_results.to_csv('output/charts/multi_model_balanced_f1_means.csv')
            print(balanced_mean_results)
            print("\nMean Balanced F1 scores saved to output/charts/multi_model_balanced_f1_means.csv")

if __name__ == "__main__":
    main() 