import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_final_paper_visuals(data_path, output_dir, baseline_algo_name='DAP'):
    """
    Generates final paper visuals by comparing the user's designed algorithms
    (DAP-A, DAP-U, G-DAP) against a single baseline (DAP).

    Args:
        data_path (str): Path to the raw experimental data CSV.
        output_dir (str): Directory to save the plots and tables.
        baseline_algo_name (str): The abbreviation for the baseline algorithm.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {data_path}")
        return

    # --- Data Preprocessing ---
    # Explicitly define which are the designed algorithms and which is the baseline
    algorithm_map = {
        'tsp150_attention_custom_final_best': 'DAP (Baseline)',
        'tsp150_finetuned_adaptive': 'DAP-A',
        'tsp150_unified_trained': 'DAP-U',
        'tsp150_uniform_pretrained': 'G-DAP'
    }
    df['Algorithm'] = df['Algorithm'].replace(algorithm_map)
    
    # Define C (Data Collection Integrity) and E (Energy Efficiency)
    df['Data Collection Integrity (C)'] = df['Packets Received'] / df['Packets Expected']
    df['Energy Efficiency (E)'] = df.apply(
        lambda row: row['Packets Received'] / row['Energy Consumed (J)'] if row['Energy Consumed (J)'] > 0 else 0,
        axis=1
    )

    # --- Data Aggregation ---
    agg_df = df.groupby(['Scenario', 'Algorithm']).mean(numeric_only=True).reset_index()

    # --- Generate LaTeX Table ---
    table_df = agg_df.pivot(index='Scenario', columns='Algorithm', values=['Data Collection Integrity (C)', 'Energy Efficiency (E)'])
    
    # Correctly order columns for clear comparison against the baseline
    col_order = [
        ('Data Collection Integrity (C)', 'DAP (Baseline)'),
        ('Data Collection Integrity (C)', 'DAP-A'),
        ('Data Collection Integrity (C)', 'DAP-U'),
        ('Data Collection Integrity (C)', 'G-DAP'),
        ('Energy Efficiency (E)', 'DAP (Baseline)'),
        ('Energy Efficiency (E)', 'DAP-A'),
        ('Energy Efficiency (E)', 'DAP-U'),
        ('Energy Efficiency (E)', 'G-DAP'),
    ]
    table_df = table_df[col_order]
    
    latex_table = table_df.to_latex(
        float_format="%.2f",
        caption="Performance Comparison of Designed Algorithms Against the Baseline.",
        label="tab:final_static_results",
        multicolumn_format='c',
        longtable=False,
        escape=False
    )
    
    table_path = os.path.join(output_dir, 'final_summary_table.tex')
    with open(table_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved final LaTeX table to {table_path}")

    # --- Relative Performance Calculation (with instance-level variance) ---
    # First, calculate the mean performance of the baseline for each scenario
    baseline_means = df[df['Algorithm'] == 'DAP (Baseline)'].groupby('Scenario').agg(
        C_baseline_mean=('Data Collection Integrity (C)', 'mean'),
        E_baseline_mean=('Energy Efficiency (E)', 'mean')
    ).reset_index()

    # Merge these baseline means back into the original dataframe
    merged_df = df.merge(baseline_means, on='Scenario')

    # Calculate improvement for each individual run relative to the baseline's mean
    # A value of 0 for the baseline mean would cause a division by zero.
    merged_df['Improvement in C (%)'] = merged_df.apply(
        lambda row: (row['Data Collection Integrity (C)'] - row['C_baseline_mean']) / row['C_baseline_mean'] * 100 if row['C_baseline_mean'] > 0 else 0,
        axis=1
    )
    merged_df['Improvement in E (%)'] = merged_df.apply(
        lambda row: (row['Energy Efficiency (E)'] - row['E_baseline_mean']) / row['E_baseline_mean'] * 100 if row['E_baseline_mean'] > 0 else 0,
        axis=1
    )

    # Plot only the user's designed algorithms
    plot_df = merged_df[merged_df['Algorithm'] != 'DAP (Baseline)']

    # --- Generate Bar Plot for Key Metrics (IEEE INFOCOM style with Error Bars) ---
    sns.set_theme(style="ticks")
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except RuntimeError:
        print("Warning: Times New Roman font not found. Using default font.")

    for metric in ['Improvement in C (%)', 'Improvement in E (%)']:
        plt.figure(figsize=(10, 6))
        
        palette = sns.color_palette("viridis", n_colors=len(plot_df['Algorithm'].unique()))
        
        # The barplot will now automatically compute the mean and confidence interval (error bars)
        ax = sns.barplot(x='Scenario', y=metric, hue='Algorithm', data=plot_df, palette=palette, errorbar=('ci', 95))
        
        # --- Polish Title and Labels for Publication ---
        title_metric = 'Data Collection Integrity (C)' if 'C' in metric else 'Energy Efficiency (E)'
        ax.set_title(f'Performance Improvement in {title_metric}\n(Relative to DAP Baseline)', fontsize=16, weight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=14, weight='bold')
        ax.set_xlabel('Scenario', fontsize=14, weight='bold')
        
        # --- Grid, Legend, and Ticks ---
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.legend(title='Designed Algorithms', fontsize=11, title_fontsize=12, loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
        ax.set_axisbelow(True)
        sns.despine(trim=True)

        plt.tight_layout()
        
        # --- Save Figures in Multiple Formats ---
        title_short = 'C' if 'C' in metric else 'E'
        output_filename_png = os.path.join(output_dir, f'final_relative_improvement_{title_short}.png')
        output_filename_pdf = os.path.join(output_dir, f'final_relative_improvement_{title_short}.pdf')
        
        plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_filename_pdf, bbox_inches='tight')
        
        print(f"Saved final key metric plot to {output_filename_png} and {output_filename_pdf}")
        plt.close()

if __name__ == '__main__':
    DATA_FILE = 'showcases/Simulating_a_data_collection_scenario_static/raw_experimental_data_batch.csv'
    OUTPUT_DIR = 'showcases/Simulating_a_data_collection_scenario_static/plots'
    generate_final_paper_visuals(DATA_FILE, output_dir=OUTPUT_DIR)