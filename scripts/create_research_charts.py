#!/usr/bin/env python3
"""
Script to create publication-quality charts for GPT city analysis across all data sources.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_gpt_data(data_source):
    """Load GPT data for specified source, comparing with research_summary.csv if it exists."""
    file_paths = {
        "reddit": "output/reddit/gpt4/classified_comments_reddit_all_gpt4_reddit_flags.csv",
        "news": "output/news/gpt4/classified_comments_news_all_gpt4_news_flags.csv", 
        "x": "output/x/gpt4/classified_comments_x_all_gpt4_x_flags.csv",
        "meeting_minutes": "output/meeting_minutes/gpt4/classified_comments_meeting_minutes_all_gpt4_meeting_minutes_flags.csv"
    }
    
    if data_source not in file_paths:
        raise ValueError(f"Unknown data source: {data_source}")
    
    flag_file_path = file_paths[data_source]
    research_summary_path = f"output/charts/gpt_research_analysis/{data_source}/research_summary_table.csv"
    
    # Check if research_summary.csv exists
    research_summary_exists = Path(research_summary_path).exists()
    flag_file_exists = Path(flag_file_path).exists()
    
    if not flag_file_exists and not research_summary_exists:
        print(f"⚠️  No data files found for {data_source}")
        return None
    
    # Compare file sizes if both exist
    if research_summary_exists and flag_file_exists:
        try:
            # Load flag file and count unique cities
            flag_df = pd.read_csv(flag_file_path)
            flag_unique_cities = flag_df['City'].nunique()
            flag_total_records = len(flag_df)
            
            # Sum the "Total Comments" column (2nd column) in research summary
            summary_df = pd.read_csv(research_summary_path)
            summary_total_comments = summary_df.iloc[:, 1].sum()  # Sum the 2nd column
            
            print(f"📊 {data_source}: Flag file has {flag_total_records} records ({flag_unique_cities} unique cities), Research summary total comments: {summary_total_comments}")
            
            # Check if flag file has more categories (for combined analysis)
            flag_categories = len([col for col in flag_df.columns if col.startswith(('Comment_', 'Critique_', 'Response_', 'Perception_', 'Racist_'))])
            summary_categories = len([col for col in summary_df.columns if col.endswith('(%)')])
            
            print(f"📊 {data_source}: Flag file has {flag_categories} categories, Research summary has {summary_categories} categories")
            
            # Prioritize flag file if it has more categories (needed for combined analysis)
            if flag_categories > summary_categories:
                print(f"✅ Using flag file (more categories: {flag_categories} vs {summary_categories})")
                return flag_df
            elif summary_total_comments > flag_total_records:
                print(f"✅ Using research summary file (more data: {summary_total_comments} vs {flag_total_records})")
                return summary_df
            else:
                # Use flag file if equal or if flag file has more data
                if summary_total_comments == flag_total_records:
                    print(f"✅ Using flag file (equal data: {flag_total_records} records each)")
                else:
                    print(f"✅ Using flag file (more data: {flag_total_records} vs {summary_total_comments})")
                return flag_df
                
        except Exception as e:
            print(f"⚠️  Error comparing files for {data_source}: {e}")
            # Fall back to flag file if comparison fails
            try:
                df = pd.read_csv(flag_file_path)
                print(f"✅ Loaded {len(df)} records from flag file for {data_source}")
                return df
            except FileNotFoundError:
                print(f"⚠️  Flag file not found: {flag_file_path}")
                return None
    
    # If only one file exists, use that one
    if flag_file_exists:
        try:
            df = pd.read_csv(flag_file_path)
            print(f"✅ Loaded {len(df)} records from flag file for {data_source}")
            return df
        except FileNotFoundError:
            print(f"⚠️  Flag file not found: {flag_file_path}")
            return None
    
    if research_summary_exists:
        try:
            df = pd.read_csv(research_summary_path)
            print(f"✅ Loaded {len(df)} records from research summary for {data_source}")
            return df
        except FileNotFoundError:
            print(f"⚠️  Research summary file not found: {research_summary_path}")
            return None
    
    return None

def extract_city_stats(df):
    """Extract statistics for each city from flags data."""
    cities = df['City'].unique()
    city_stats = []
    
    for city in cities:
        city_data = df[df['City'] == city]
        total_comments = len(city_data)
        
        # Find all flag columns (excluding City and Comment)
        flag_columns = [col for col in df.columns if col not in ['City', 'Comment']]
        
        # Calculate percentages for each flag category
        flag_stats = {}
        for col in flag_columns:
            if col in city_data.columns:
                flag_count = (city_data[col] != 0).sum()
                flag_percentage = (flag_count / total_comments * 100) if total_comments > 0 else 0
                flag_stats[col] = flag_percentage
        
        city_stat = {
            'City': city,
            'Total_Comments': total_comments
        }
        city_stat.update(flag_stats)
        
        city_stats.append(city_stat)
    
    return pd.DataFrame(city_stats)

def create_publication_charts(df_stats, data_source="reddit"):
    """Create publication-quality charts."""
    # Set up publication-quality styling
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Use a clean, professional font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    # Create output directory
    output_dir = Path(f"output/charts/gpt_research_analysis/{data_source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Creating charts in: {output_dir}")
    
    # Add city size grouping based on actual population size
    large_cities_list = ['san francisco', 'portland', 'buffalo', 'baltimore', 'el paso']
    small_cities_list = ['kalamazoo', 'south bend', 'rockford', 'scranton', 'fayetteville']
    
    df_stats['City_Size'] = 'Small'
    df_stats.loc[df_stats['City'].str.lower().isin(large_cities_list), 'City_Size'] = 'Large'
    
    # Calculate group statistics
    large_cities = df_stats[df_stats['City_Size'] == 'Large']
    small_cities = df_stats[df_stats['City_Size'] == 'Small']
    
    print(f"Large cities: {len(large_cities)} ({', '.join(large_cities['City'].tolist())})")
    print(f"Small cities: {len(small_cities)} ({', '.join(small_cities['City'].tolist())})")
    
    # Sort by total comments
    df_sorted = df_stats.sort_values('Total_Comments', ascending=False)
    
    # Use only subcategories for all group comparisons and significance testing
    exclude_cols = ['Total_Comments', 'City_Size', 'Comment Type', 'Critique Category', 
                   'Response Category', 'Perception Type', 'racist', 'Reasoning', 'Raw Response']
    flag_columns = [col for col in df_stats.columns if col not in ['City', 'Total_Comments', 'City_Size']]
    subcategory_cols = [col for col in flag_columns if col not in exclude_cols]
    


    # Figure 5: Heatmap of all categories
    # Prepare data for heatmap
    heatmap_data = df_sorted.copy()
    heatmap_data.set_index('City', inplace=True)
    
    # Select only subcategory columns (exclude main categories and metadata)
    exclude_cols = ['Total_Comments', 'City_Size', 'Comment Type', 'Critique Category', 
                   'Response Category', 'Perception Type', 'racist', 'Reasoning', 'Raw Response']
    subcategory_cols = [col for col in heatmap_data.columns if col not in exclude_cols]
    heatmap_data = heatmap_data[subcategory_cols]
    
    # Rename columns for better readability (remove category prefixes)
    clean_columns = []
    for col in heatmap_data.columns:
        clean_col = col
        for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
            if col.startswith(prefix):
                clean_col = col.replace(prefix, '')
                break
        # Handle Racist_Flag specifically
        if col == 'Racist_Flag':
            clean_col = 'Racist'
        else:
            clean_col = clean_col.replace('_', ' ').title()
        clean_columns.append(clean_col)
    heatmap_data.columns = clean_columns
    
    fig5, ax5 = plt.subplots(1, 1, figsize=(14, 8))
    sns.heatmap(heatmap_data.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Percentage (%)'}, ax=ax5)
    ax5.set_title(f'{data_source.title()} Classification Categories Heatmap by City', fontweight='bold', pad=20)
    ax5.set_xlabel('City', fontweight='bold')
    ax5.set_ylabel('Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 7: Large vs Small Cities Comparison
    large_means = large_cities[subcategory_cols].mean()
    small_means = small_cities[subcategory_cols].mean()
    x = np.arange(len(subcategory_cols))
    width = 0.35
    fig7, ax7 = plt.subplots(1, 1, figsize=(max(12, len(subcategory_cols)*0.7), 8))
    bars1 = ax7.bar(x - width/2, large_means, width, label='Large Cities', 
                     color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax7.bar(x + width/2, small_means, width, label='Small Cities', 
                     color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax7.set_title(f'{data_source.title()}: Large vs Small Cities Classification Patterns', fontweight='bold', pad=20)
    ax7.set_ylabel('Average Percentage (%)', fontweight='bold')
    ax7.set_xlabel('Classification Subcategory', fontweight='bold')
    ax7.set_xticks(x)
    # Clean subcategory labels for x-axis
    clean_labels = []
    for col in subcategory_cols:
        clean_label = col
        for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
            if col.startswith(prefix):
                clean_label = col.replace(prefix, '')
                break
        # Handle Racist_Flag specifically
        if col == 'Racist_Flag':
            clean_label = 'Racist'
        else:
            clean_label = clean_label.replace('_', ' ').title()
        clean_labels.append(clean_label)
    ax7.set_xticklabels(clean_labels, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure7_city_size_comparison.pdf', bbox_inches='tight')
    plt.close()

    # Figure 8: Confusion Matrix of Category Relationships
    fig8, ax8 = plt.subplots(1, 1, figsize=(max(12, len(subcategory_cols)*0.7), 10))
    correlation_matrix = df_stats[subcategory_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax8, square=True)
    ax8.set_title(f'{data_source.title()} Category Relationship Matrix', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure8_category_correlation.pdf', bbox_inches='tight')
    plt.close()

    # Figure 9: Statistical Significance Test Results
    from scipy import stats
    bonferroni_alpha = 0.05 / 16  # Fixed Bonferroni correction for 16 comparisons
    significant_differences = []
    p_values = []
    for col in subcategory_cols:
        large_data = large_cities[col].values
        small_data = small_cities[col].values
        t_stat, p_val = stats.ttest_ind(large_data, small_data)
        p_values.append(p_val)
        if p_val < bonferroni_alpha:
            significant_differences.append(col)
    x_pos = np.arange(len(subcategory_cols))
    colors = ['red' if p < bonferroni_alpha else 'gray' for p in p_values]
    fig9, ax9 = plt.subplots(1, 1, figsize=(max(12, len(subcategory_cols)*0.7), 8))
    bars = ax9.bar(x_pos, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
    ax9.axhline(y=-np.log10(bonferroni_alpha), color='red', linestyle='--', label=f'Bonferroni α={bonferroni_alpha:.4f}')
    ax9.set_title(f'{data_source.title()} Statistical Significance: Large vs Small Cities', fontweight='bold', pad=20)
    ax9.set_ylabel('-log10(p-value)', fontweight='bold')
    ax9.set_xlabel('Classification Subcategory', fontweight='bold')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(clean_labels, rotation=45, ha='right')
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        if p_val < bonferroni_alpha:
            ax9.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'p={p_val:.3g}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'figure9_significance_test.pdf', bbox_inches='tight')
    plt.close()

    print(f"📊 Created 4 publication-quality figures (PDF only) for {data_source}:")
    print("  - Figure 5: Classification Categories Heatmap")
    print("  - Figure 7: Large vs Small Cities Comparison")
    print("  - Figure 8: Category Relationship Matrix")
    print("  - Figure 9: Statistical Significance Test")
    if significant_differences:
        print(f"\n🔍 Statistically significant differences (Bonferroni α={bonferroni_alpha:.4f}) between large and small cities:")
        for diff in significant_differences:
            print(f"  - {diff.replace('_', ' ').title()}")
    else:
        print(f"\n🔍 No statistically significant differences found (Bonferroni α={bonferroni_alpha:.4f}).")
    
    return df_stats

def create_summary_table(df_stats, data_source="reddit"):
    """Create a summary table for the paper."""
    print(f"\n📋 Summary Statistics for {data_source.title()} Research Paper:")
    print("=" * 80)
    
    # Sort by total comments
    df_sorted = df_stats.sort_values('Total_Comments', ascending=False)
    
    # Check if this is already a research summary table (has cleaned column names)
    if 'Express Opinion (%)' in df_stats.columns:
        # Already a research summary table, just display it
        print(df_sorted.to_string(index=False, float_format='%.1f'))
        
        # Save summary table
        output_dir = Path(f"output/charts/gpt_research_analysis/{data_source}")
        df_sorted.to_csv(output_dir / 'research_summary_table.csv', index=False)
        print(f"\n📄 Summary table saved to: {output_dir}/research_summary_table.csv")
        return
    
    # Create a clean summary table from flag data
    summary_cols = ['City', 'Total_Comments', 'Comment_express their opinion', 
                   'Critique_societal critique', 'Response_solutions/interventions',
                   'Perception_deserving/undeserving', 'Racist_Flag']
    
    # Check which columns exist in the dataframe
    available_cols = [col for col in summary_cols if col in df_sorted.columns]
    
    if len(available_cols) < 2:  # Need at least City and Total_Comments
        print(f"❌ Error: Required columns not found in {data_source} data")
        print(f"Available columns: {list(df_sorted.columns)}")
        return
    
    summary_df = df_sorted[available_cols].copy()
    
    # Map column names to display names
    column_mapping = {
        'City': 'City',
        'Total_Comments': 'Total Comments',
        'Comment_express their opinion': 'Express Opinion (%)',
        'Critique_societal critique': 'Societal Critique (%)',
        'Response_solutions/interventions': 'Solutions (%)',
        'Perception_deserving/undeserving': 'Deserving/Undeserving (%)',
        'Racist_Flag': 'Racist (%)'
    }
    
    # Rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in summary_df.columns:
            summary_df = summary_df.rename(columns={old_col: new_col})
    
    print(summary_df.to_string(index=False, float_format='%.1f'))
    
    # Save summary table
    output_dir = Path(f"output/charts/gpt_research_analysis/{data_source}")
    summary_df.to_csv(output_dir / 'research_summary_table.csv', index=False)
    print(f"\n📄 Summary table saved to: {output_dir}/research_summary_table.csv")

def process_data_source(data_source):
    """Process a single data source."""
    print(f"\n🔬 Processing {data_source.upper()} data")
    print("=" * 50)
    
    # Load data
    df = load_gpt_data(data_source)
    if df is None:
        print(f"❌ Skipping {data_source} - data not available")
        return None
    
    # Check if this is already a research summary table (has cleaned column names)
    if 'Express Opinion (%)' in df.columns:
        print(f"✅ Using existing research summary table for {data_source}")
        # This is already aggregated data, use it directly
        df_stats = df.copy()
        # Ensure it has the expected column name for Total_Comments
        if 'Total Comments' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Total Comments': 'Total_Comments'})
    else:
        # Extract city statistics from raw flag data
        df_stats = extract_city_stats(df)
    
    # If this is a research summary table, map the cleaned column names back to flag names for chart processing
    if 'Express Opinion (%)' in df_stats.columns:
        print(f"🔄 Mapping research summary column names to flag format for {data_source}")
        # Map research summary columns to expected flag column names
        column_mapping = {
            'Express Opinion (%)': 'Comment_express their opinion',
            'Societal Critique (%)': 'Critique_societal critique', 
            'Solutions (%)': 'Response_solutions/interventions',
            'Deserving/Undeserving (%)': 'Perception_deserving/undeserving',
            'Racist (%)': 'Racist_Flag'
        }
        
        # Create a copy with mapped column names for chart processing
        df_stats_for_charts = df_stats.copy()
        for clean_name, flag_name in column_mapping.items():
            if clean_name in df_stats_for_charts.columns:
                df_stats_for_charts = df_stats_for_charts.rename(columns={clean_name: flag_name})
        
        # Use the mapped dataframe for chart creation
        df_stats = df_stats_for_charts
    
    # Create publication charts
    df_stats = create_publication_charts(df_stats, data_source)
    
    # Create summary table
    create_summary_table(df_stats, data_source)
    
    return df_stats

def create_combined_analysis(results):
    """Create combined analysis across all data sources with weighted averaging."""
    print("\n🔬 Creating Combined Analysis Across All Data Sources")
    print("=" * 80)
    
    # Filter out None results
    valid_results = {source: stats for source, stats in results.items() if stats is not None}
    
    if len(valid_results) < 2:
        print("❌ Need at least 2 data sources for combined analysis")
        return None
    
    # Add source column to each dataframe
    combined_stats = []
    for source, df_stats in valid_results.items():
        df_stats_copy = df_stats.copy()
        df_stats_copy['Source'] = source
        combined_stats.append(df_stats_copy)
    
    # Combine all stats
    combined_df = pd.concat(combined_stats, ignore_index=True)
    print(f"📊 Combined dataset: {len(combined_df)} city-source combinations")
    
    # Create output directory
    output_dir = Path("output/charts/gpt_research_analysis/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Creating combined charts in: {output_dir}")
    
    # Set up publication-quality styling
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    # Get subcategory columns
    exclude_cols = ['Total_Comments', 'Source', 'City_Size', 'Comment Type', 'Critique Category', 
                   'Response Category', 'Perception Type', 'racist', 'Reasoning', 'Raw Response']
    flag_columns = [col for col in combined_df.columns if col not in ['City', 'Total_Comments', 'Source', 'City_Size']]
    subcategory_cols = [col for col in flag_columns if col not in exclude_cols]
    
    # Figure 1: Data Source Distribution
    source_counts = combined_df.groupby('Source')['Total_Comments'].sum()
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax1.bar(source_counts.index, source_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Total Comments by Data Source', fontweight='bold', pad=20)
    ax1.set_ylabel('Number of Comments', fontweight='bold')
    ax1.set_xlabel('Data Source', fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(source_counts.values)*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_data_source_distribution.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Weighted Average by Source
    # Calculate weighted averages for each source
    weighted_avgs = {}
    for source in valid_results.keys():
        source_data = combined_df[combined_df['Source'] == source]
        # Weight by number of comments
        weights = source_data['Total_Comments']
        weighted_avg = {}
        for col in subcategory_cols:
            if col in source_data.columns:
                weighted_avg[col] = np.average(source_data[col], weights=weights)
            else:
                # If category doesn't exist for this source, set to 0
                weighted_avg[col] = 0.0
        weighted_avgs[source] = weighted_avg
    
    # Create comparison chart
    sources = list(weighted_avgs.keys())
    categories = list(weighted_avgs[sources[0]].keys()) if sources else []
    
    x = np.arange(len(categories))
    width = 0.2
    fig2, ax2 = plt.subplots(1, 1, figsize=(max(15, len(categories)*0.8), 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for i, source in enumerate(sources):
        values = [weighted_avgs[source].get(cat, 0) for cat in categories]
        bars = ax2.bar(x + i*width, values, width, label=source.title(), 
                       color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., val + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax2.set_title('Weighted Average Classification by Data Source', fontweight='bold', pad=20)
    ax2.set_ylabel('Weighted Average Percentage (%)', fontweight='bold')
    ax2.set_xlabel('Classification Category', fontweight='bold')
    ax2.set_xticks(x + width * (len(sources)-1)/2)
    
    # Clean category labels
    clean_labels = []
    for cat in categories:
        clean_label = cat
        for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
            if cat.startswith(prefix):
                clean_label = cat.replace(prefix, '')
                break
        if cat == 'Racist_Flag':
            clean_label = 'Racist'
        else:
            clean_label = clean_label.replace('_', ' ').title()
        clean_labels.append(clean_label)
    
    ax2.set_xticklabels(clean_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_weighted_averages.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 3: Overall Weighted Average (Combined)
    # Calculate overall weighted average across all sources
    total_comments = combined_df['Total_Comments'].sum()
    overall_weighted_avg = {}
    
    for col in subcategory_cols:
        if col in combined_df.columns:
            # Weight by number of comments
            weighted_sum = (combined_df[col] * combined_df['Total_Comments']).sum()
            overall_weighted_avg[col] = weighted_sum / total_comments if total_comments > 0 else 0
    
    # Create overall average chart
    categories = list(overall_weighted_avg.keys())
    values = list(overall_weighted_avg.values())
    
    fig3, ax3 = plt.subplots(1, 1, figsize=(max(12, len(categories)*0.6), 8))
    bars = ax3.bar(range(len(categories)), values, color='#2E86AB', alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    ax3.set_title('Overall Weighted Average Classification (All Sources)', fontweight='bold', pad=20)
    ax3.set_ylabel('Weighted Average Percentage (%)', fontweight='bold')
    ax3.set_xlabel('Classification Category', fontweight='bold')
    ax3.set_xticks(range(len(categories)))
    
    # Clean category labels
    clean_labels = []
    for cat in categories:
        clean_label = cat
        for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
            if cat.startswith(prefix):
                clean_label = cat.replace(prefix, '')
                break
        if cat == 'Racist_Flag':
            clean_label = 'Racist'
        else:
            clean_label = clean_label.replace('_', ' ').title()
        clean_labels.append(clean_label)
    
    ax3.set_xticklabels(clean_labels, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., val + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_overall_weighted_average.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 4: Source Comparison Heatmap
    # Create heatmap comparing sources
    heatmap_data = pd.DataFrame(weighted_avgs).T
    
    # Clean column names for display
    clean_columns = []
    for col in heatmap_data.columns:
        clean_col = col
        for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
            if col.startswith(prefix):
                clean_col = col.replace(prefix, '')
                break
        if col == 'Racist_Flag':
            clean_col = 'Racist'
        else:
            clean_col = clean_col.replace('_', ' ').title()
        clean_columns.append(clean_col)
    heatmap_data.columns = clean_columns
    
    fig4, ax4 = plt.subplots(1, 1, figsize=(max(12, len(clean_columns)*0.8), 6))
    sns.heatmap(heatmap_data.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Weighted Average (%)'}, ax=ax4)
    ax4.set_title('Data Source Comparison Heatmap', fontweight='bold', pad=20)
    ax4.set_xlabel('Data Source', fontweight='bold')
    ax4.set_ylabel('Classification Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_source_comparison_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 4.5: Comprehensive Source Comparison with Significance Testing
    # Calculate means and standard errors for all categories across sources
    from scipy import stats
    
    # Prepare data for all 16 categories
    all_categories = subcategory_cols
    source_means = {}
    source_errors = {}
    significant_categories = []
    
    # Calculate means and standard errors for each source and category
    for source in sources:
        source_data = combined_df[combined_df['Source'] == source]
        source_means[source] = {}
        source_errors[source] = {}
        
        for cat in all_categories:
            if cat in source_data.columns:
                values = source_data[cat].values
                mean_val = np.mean(values)
                std_err = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                source_means[source][cat] = mean_val
                source_errors[source][cat] = std_err
    
    # Perform ANOVA and pairwise tests for each category
    bonferroni_alpha = 0.05 / 16  # Fixed Bonferroni correction for 16 comparisons
    p_values = []
    pairwise_significance = {}  # Store which pairs are significant for each category
    
    for cat in all_categories:
        if cat in combined_df.columns:
            # Group data by source
            source_data = [combined_df[combined_df['Source'] == source][cat].values 
                         for source in sources if source in combined_df['Source'].values]
            
            if len(source_data) > 1 and all(len(data) > 0 for data in source_data):
                # Perform one-way ANOVA
                f_stat, p_val = stats.f_oneway(*source_data)
                p_values.append(p_val)
                
                if p_val < bonferroni_alpha:
                    significant_categories.append(cat)
                
                # Perform pairwise t-tests
                pairwise_significance[cat] = {}
                for i, source1 in enumerate(sources):
                    for j, source2 in enumerate(sources):
                        if i < j:  # Only test each pair once
                            data1 = combined_df[(combined_df['Source'] == source1) & (combined_df[cat].notna())][cat].values
                            data2 = combined_df[(combined_df['Source'] == source2) & (combined_df[cat].notna())][cat].values
                            
                            if len(data1) > 0 and len(data2) > 0:
                                t_stat, p_val_pair = stats.ttest_ind(data1, data2)
                                pairwise_significance[cat][(source1, source2)] = p_val_pair < bonferroni_alpha
                            else:
                                pairwise_significance[cat][(source1, source2)] = False
            else:
                p_values.append(1.0)
                pairwise_significance[cat] = {}
    
    # Clean category labels for display
    clean_labels = []
    for cat in all_categories:
        clean_label = cat
        for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
            if cat.startswith(prefix):
                clean_label = cat.replace(prefix, '')
                break
        if cat == 'Racist_Flag':
            clean_label = 'Racist'
        else:
            clean_label = clean_label.replace('_', ' ').title()
        clean_labels.append(clean_label)
    
    # Create comprehensive comparison chart with subplots
    x = np.arange(len(all_categories))
    width = 0.2
    fig4_5, (ax4_5, ax_pairwise) = plt.subplots(2, 1, figsize=(20, 16), 
                                                   gridspec_kw={'height_ratios': [3, 1]})
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    alpha_normal = 0.7
    alpha_significant = 0.9
    
    for i, source in enumerate(sources):
        means = [source_means[source].get(cat, 0) for cat in all_categories]
        errors = [source_errors[source].get(cat, 0) for cat in all_categories]
        
        # Determine which categories are significant for this source
        significant_mask = [cat in significant_categories for cat in all_categories]
        
        # Create bars with different alpha for significant categories
        bars = ax4_5.bar(x + i*width, means, width, label=source.title(), 
                         color=colors[i], alpha=alpha_significant if any(significant_mask) else alpha_normal,
                         edgecolor='black', linewidth=0.5)
        
        # Add error bars
        ax4_5.errorbar(x + i*width, means, yerr=errors, fmt='none', 
                       color='black', capsize=3, capthick=1, linewidth=1)
        
        # Add value labels on bars
        for j, (bar, mean_val, error_val) in enumerate(zip(bars, means, errors)):
            if mean_val > 0:
                # Use different formatting for significant categories
                if significant_mask[j]:
                    ax4_5.text(bar.get_x() + bar.get_width()/2., mean_val + error_val + 0.5,
                              f'{mean_val:.1f}%', ha='center', va='bottom', 
                              fontsize=8, fontweight='bold', color='red')
                else:
                    ax4_5.text(bar.get_x() + bar.get_width()/2., mean_val + error_val + 0.5,
                              f'{mean_val:.1f}%', ha='center', va='bottom', 
                              fontsize=8)
    
    # Create pairwise significance chart in subplot
    if significant_categories:
        # Prepare data for pairwise significance heatmap
        pairwise_data = []
        num_pairs = len(sources) * (len(sources) - 1) // 2  # Number of unique pairs
        
        for cat in all_categories:
            row = [0] * num_pairs  # Initialize with zeros
            if cat in significant_categories and cat in pairwise_significance:
                pair_idx = 0
                for i, source1 in enumerate(sources):
                    for j, source2 in enumerate(sources):
                        if i < j:  # Only upper triangle
                            is_sig = pairwise_significance[cat].get((source1, source2), False)
                            row[pair_idx] = 1 if is_sig else 0
                            pair_idx += 1
            pairwise_data.append(row)
        
        # Create pairwise significance heatmap
        pairwise_array = np.array(pairwise_data)
        
        # Create custom colormap for significance
        from matplotlib.colors import ListedColormap
        colors_pairwise = ['white', 'red']
        cmap_pairwise = ListedColormap(colors_pairwise)
        
        # Create mask for upper triangle only
        mask = np.zeros_like(pairwise_array, dtype=bool)
        for i in range(pairwise_array.shape[1]):
            for j in range(pairwise_array.shape[1]):
                if i >= j:  # Mask lower triangle and diagonal
                    mask[:, i] = True
        
        # Plot pairwise significance heatmap
        im = ax_pairwise.imshow(pairwise_array.T, cmap=cmap_pairwise, aspect='auto', 
                               vmin=0, vmax=1, alpha=0.8)
        
        # Set up axes
        ax_pairwise.set_xticks(range(len(all_categories)))
        ax_pairwise.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=8, fontweight='bold')
        ax_pairwise.set_yticks(range(num_pairs))
        
        # Create pairwise labels
        pairwise_labels = []
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                if i < j:
                    pairwise_labels.append(f'{source1.title()}-{source2.title()}')
        ax_pairwise.set_yticklabels(pairwise_labels, fontsize=8, fontweight='bold')
        
        ax_pairwise.set_title('Pairwise Significance Between Data Sources\n' + 
                            f'(Red = Significant, Bonferroni α={bonferroni_alpha:.4f})', 
                            fontweight='bold', pad=10, fontsize=10)
        
        # Add grid
        ax_pairwise.grid(True, alpha=0.3)
    else:
        # If no significant categories, show empty plot
        ax_pairwise.text(0.5, 0.5, 'No Significant Pairwise Differences', 
                        ha='center', va='center', transform=ax_pairwise.transAxes,
                        fontsize=12, fontweight='bold', color='gray')
        ax_pairwise.set_title('Pairwise Significance Between Data Sources', 
                            fontweight='bold', pad=10, fontsize=10)
    
    # Add significance indicators
    if significant_categories:
        # Add asterisks above significant categories
        for j, cat in enumerate(all_categories):
            if cat in significant_categories:
                max_height = max([source_means[source].get(cat, 0) + source_errors[source].get(cat, 0) 
                                for source in sources])
                ax4_5.text(x[j] + width * 1.5, max_height + 2, '***', 
                           ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
    
    ax4_5.set_title('Comprehensive Comparison: All Categories Across Data Sources\n' + 
                    f'(*** = Overall Significant, Bonferroni α={bonferroni_alpha:.4f})', 
                    fontweight='bold', pad=20, fontsize=14)
    ax4_5.set_ylabel('Weighted Average Percentage (%)', fontweight='bold', fontsize=12)
    ax4_5.set_xlabel('Classification Category', fontweight='bold', fontsize=12)
    ax4_5.set_xticks(x + width * 1.5)
    ax4_5.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax4_5.legend(fontsize=11)
    ax4_5.grid(axis='y', alpha=0.3)
    
    # Add a subtle background color for significant categories
    if significant_categories:
        for j, cat in enumerate(all_categories):
            if cat in significant_categories:
                ax4_5.axvspan(x[j] - width/2, x[j] + width*3.5, alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_5_comprehensive_source_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 5: Statistical Significance Testing
    # Test for significant differences between sources
    if len(sources) > 1:
        from scipy import stats
        # Perform ANOVA for each category with Bonferroni correction
        significant_categories = []
        p_values = []
        bonferroni_alpha = 0.05 / 16  # Fixed Bonferroni correction for 16 comparisons
        
        for col in subcategory_cols:
            if col in combined_df.columns:
                # Group data by source
                source_data = [combined_df[combined_df['Source'] == source][col].values 
                             for source in sources if source in combined_df['Source'].values]
                
                if len(source_data) > 1 and all(len(data) > 0 for data in source_data):
                    # Perform one-way ANOVA
                    f_stat, p_val = stats.f_oneway(*source_data)
                    p_values.append(p_val)
                    
                    if p_val < bonferroni_alpha:
                        significant_categories.append(col)
                else:
                    p_values.append(1.0)  # No significant difference
        
        # Create significance chart
        x_pos = np.arange(len(subcategory_cols))
        colors = ['red' if p < bonferroni_alpha else 'gray' for p in p_values]
        
        fig5, ax5 = plt.subplots(1, 1, figsize=(max(12, len(subcategory_cols)*0.7), 8))
        bars = ax5.bar(x_pos, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
        ax5.axhline(y=-np.log10(bonferroni_alpha), color='red', linestyle='--', label=f'Bonferroni α={bonferroni_alpha:.4f}')
        ax5.set_title('Statistical Significance: Differences Between Data Sources', fontweight='bold', pad=20)
        ax5.set_ylabel('-log10(p-value)', fontweight='bold')
        ax5.set_xlabel('Classification Category', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(clean_labels, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # Add p-value labels for significant categories
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < bonferroni_alpha:
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'p={p_val:.3g}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'figure5_significance_test.pdf', bbox_inches='tight')
        plt.close()
        
        if significant_categories:
            print(f"\n🔍 Statistically significant differences (Bonferroni α={bonferroni_alpha:.4f}) between data sources:")
            for cat in significant_categories:
                clean_cat = cat
                for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
                    if cat.startswith(prefix):
                        clean_cat = cat.replace(prefix, '')
                        break
                if cat == 'Racist_Flag':
                    clean_cat = 'Racist'
                else:
                    clean_cat = clean_cat.replace('_', ' ').title()
                print(f"  - {clean_cat}")
        else:
            print(f"\n🔍 No statistically significant differences found between data sources (Bonferroni α={bonferroni_alpha:.4f}).")
    
    print(f"\n📊 Created 6 combined analysis figures:")
    print("  - Figure 1: Data Source Distribution")
    print("  - Figure 2: Weighted Average by Source")
    print("  - Figure 3: Overall Weighted Average")
    print("  - Figure 4: Source Comparison Heatmap")
    print("  - Figure 4.5: Comprehensive Source Comparison (All 16 Categories)")
    if len(sources) > 1:
        print("  - Figure 5: Statistical Significance Test")
    
    return combined_df, weighted_avgs, overall_weighted_avg

def create_combined_summary_table(combined_df, weighted_avgs, overall_weighted_avg):
    """Create a comprehensive summary table for combined analysis."""
    print(f"\n📋 Combined Analysis Summary:")
    print("=" * 80)
    
    # Create summary table
    summary_data = []
    
    # Add overall weighted averages
    for category, value in overall_weighted_avg.items():
        clean_cat = category
        for prefix in ['Comment_', 'Critique_', 'Response_', 'Perception_']:
            if category.startswith(prefix):
                clean_cat = category.replace(prefix, '')
                break
        if category == 'Racist_Flag':
            clean_cat = 'Racist'
        else:
            clean_cat = clean_cat.replace('_', ' ').title()
        
        row = {'Category': clean_cat, 'Overall Weighted Avg (%)': f'{value:.1f}'}
        
        # Add source-specific averages
        for source, source_avgs in weighted_avgs.items():
            source_val = source_avgs.get(category, 0)
            row[f'{source.title()} (%)'] = f'{source_val:.1f}'
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add data source statistics
    source_stats = combined_df.groupby('Source').agg({
        'Total_Comments': ['sum', 'count'],
        'City': 'nunique'
    }).round(1)
    
    print("\n📊 Data Source Statistics:")
    print(source_stats)
    
    print("\n📊 Overall Weighted Averages by Category:")
    print(summary_df.to_string(index=False))
    
    # Save summary table
    output_dir = Path("output/charts/gpt_research_analysis/combined")
    summary_df.to_csv(output_dir / 'combined_summary_table.csv', index=False)
    print(f"\n📄 Combined summary table saved to: {output_dir}/combined_summary_table.csv")

def load_category_data():
    """Load the category overall agreement data"""
    df = pd.read_csv('output/annotation/category_overall_agreement.csv')
    return df

def create_agreement_plot():
    """Create a bar chart showing agreement rates by category"""
    df = load_category_data()
    
    # Sort by overall agreement rate for better visualization
    df = df.sort_values('Overall Agreement Rate (%)', ascending=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Colors for different agreement levels
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4ecdc4']
    
    # Plot 1: Stacked bar chart showing agreement levels (0, 1, 2, 3 annotators)
    categories = df['Category']
    no_agreement = df['No Agreement (0)']
    partial_1 = df['Partial Agreement (1-2)'] * 0.5  # Split partial agreement
    partial_2 = df['Partial Agreement (1-2)'] * 0.5
    full_agreement = df['Full Agreement (3)']
    
    # Create stacked bars
    x = np.arange(len(categories))
    width = 0.7
    
    bars1 = ax1.bar(x, no_agreement, width, label='Negative Agreement (0)', color=colors[0])
    bars2 = ax1.bar(x, partial_1, width, bottom=no_agreement, label='Partial Negative (1)', color=colors[1])
    bars3 = ax1.bar(x, partial_2, width, bottom=no_agreement+partial_1, label='Partial Positive (2)', color=colors[2])
    bars4 = ax1.bar(x, full_agreement, width, bottom=no_agreement+partial_1+partial_2, label='Positive Agreement (3)', color=colors[3])
    
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Number of Labels')
    ax1.set_title('Annotation Agreement Distribution by Category (3 Annotators)', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Remove value labels from bars - they will be shown below instead
    
    # Plot 2: Overall agreement rate by category
    agreement_rates = df['Overall Agreement Rate (%)']
    
    bars = ax2.barh(x, agreement_rates, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Agreement Rate (%)')
    ax2.set_ylabel('Categories')
    ax2.set_title('Overall Annotation Agreement Rate by Category (3 Annotators)', fontsize=16, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(categories)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, agreement_rates)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{rate:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Add a horizontal line at 50% for reference
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% Reference')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def create_simplified_plot():
    """Create a simplified version focusing on agreement rates"""
    df = load_category_data()
    
    # Sort by agreement rate
    df = df.sort_values('Overall Agreement Rate (%)', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = df['Category']
    agreement_rates = df['Overall Agreement Rate (%)']
    
    # Color bars based on agreement rate
    colors = ['red' if rate < 50 else 'orange' if rate < 70 else 'green' for rate in agreement_rates]
    
    bars = ax.barh(range(len(categories)), agreement_rates, color=colors, alpha=0.7)
    
    ax.set_xlabel('Agreement Rate (%)', fontsize=12)
    ax.set_ylabel('Categories', fontsize=12)
    ax.set_title('Annotation Agreement Rate by Category (3 Annotators)\n(Negative and Positive Agreement Combined)', 
                fontsize=16, fontweight='bold')
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, agreement_rates)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{rate:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Add reference lines
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% Reference')
    ax.axvline(x=70, color='orange', linestyle='--', alpha=0.7, label='70% Reference')
    ax.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='90% Reference')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_agreement_plots():
    """Create and save agreement rate plots"""
    print("\n📊 Creating agreement rate plots...")
    
    try:
        # Create the detailed plot
        fig1 = create_agreement_plot()
        fig1.savefig('output/annotation/agreement_by_category_detailed.pdf', 
                     bbox_inches='tight', dpi=300)
        print("✅ Detailed plot saved as: output/annotation/agreement_by_category_detailed.pdf")
        
        # Create the simplified plot
        fig2 = create_simplified_plot()
        fig2.savefig('output/annotation/agreement_by_category.pdf', 
                     bbox_inches='tight', dpi=300)
        print("✅ Simplified plot saved as: output/annotation/agreement_by_category.pdf")
        
    except FileNotFoundError:
        print("⚠️  Warning: Could not find agreement data file. Skipping agreement plots.")
        print("   Expected file: output/annotation/category_overall_agreement.csv")
    except Exception as e:
        print(f"❌ Error creating agreement plots: {e}")

def main():
    """Main function."""
    print("🔬 Creating Publication-Quality Research Charts for All Data Sources")
    print("=" * 80)
    
    # Define all data sources
    data_sources = ["reddit", "news", "x", "meeting_minutes"]
    
    # Process each data source
    results = {}
    for source in data_sources:
        try:
            results[source] = process_data_source(source)
        except Exception as e:
            print(f"❌ Error processing {source}: {e}")
            results[source] = None
    
    # Summary of individual sources
    successful_sources = [source for source, result in results.items() if result is not None]
    print(f"\n✅ Successfully processed {len(successful_sources)} data sources: {', '.join(successful_sources)}")
    
    if len(successful_sources) < len(data_sources):
        failed_sources = [source for source in data_sources if source not in successful_sources]
        print(f"⚠️  Failed to process: {', '.join(failed_sources)}")
    
    # Create combined analysis
    combined_results = create_combined_analysis(results)
    
    if combined_results is not None:
        combined_df, weighted_avgs, overall_weighted_avg = combined_results
        
        # Create combined summary table
        create_combined_summary_table(combined_df, weighted_avgs, overall_weighted_avg)
        
        print(f"\n✅ Combined analysis completed!")
        print(f"📁 Combined results saved to: output/charts/gpt_research_analysis/combined/")
    else:
        print("❌ Combined analysis could not be completed")
    
    # Create agreement plots
    create_agreement_plots()

if __name__ == "__main__":
    main() 