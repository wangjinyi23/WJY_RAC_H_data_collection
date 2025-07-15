import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_focused_visuals(data_path, output_dir):
    """
    Generates a focused set of four plots for paper publication, highlighting
    algorithm advantages and generalization.

    Args:
        data_path (str): Path to the raw experimental data CSV.
        output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {data_path}")
        return

    # --- Data Preprocessing ---
    df.columns = df.columns.str.strip()
    df['Data Collection Rate'] = df['Packets Received'] / df['Packets Expected']
    
    # Aggregate data: calculate the mean for each algorithm-scenario-devices combination
    # This simplifies the dataset for clearer bar plots.
    agg_df = df.groupby(['Algorithm', 'Scenario', 'Num Devices']).mean(numeric_only=True).reset_index()

    # --- Plotting Style ---
    sns.set_theme(style="whitegrid")
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
    except RuntimeError:
        print("Warning: Times New Roman font not found. Using default font.")

    # --- Plot 1: Path Length Comparison ---
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Scenario', y='Path Length (L)', hue='Algorithm', data=agg_df, palette='viridis')
    plt.title('Path Length Comparison Across Scenarios', fontsize=16, weight='bold')
    plt.ylabel('Average Path Length (L)', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.legend(title='Algorithm', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'focused_path_length_comparison.png'), dpi=300)
    plt.close()
    print("Saved Plot 1: Path Length Comparison")

    # --- Plot 2: Data Collection Rate Comparison ---
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Scenario', y='Data Collection Rate', hue='Algorithm', data=agg_df, palette='viridis')
    plt.title('Data Collection Rate Comparison Across Scenarios', fontsize=16, weight='bold')
    plt.ylabel('Average Data Collection Rate', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylim(0, 1) # Data collection rate is a percentage
    plt.legend(title='Algorithm', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'focused_data_collection_rate.png'), dpi=300)
    plt.close()
    print("Saved Plot 2: Data Collection Rate Comparison")

    # --- Plot 3: Relative Improvement vs. LKH ---
    baseline_algo = 'LKH'
    # Calculate mean path length for LKH in each scenario and for each device number
    baseline_metrics = agg_df[agg_df['Algorithm'] == baseline_algo].set_index(['Scenario', 'Num Devices'])['Path Length (L)']
    
    # Create a temporary dataframe for comparison
    comparison_df = agg_df.copy()
    comparison_df = comparison_df.join(baseline_metrics, on=['Scenario', 'Num Devices'], rsuffix='_baseline')

    # Calculate relative improvement
    comparison_df['Path Length Improvement (%)'] = (
        (comparison_df['Path Length (L)_baseline'] - comparison_df['Path Length (L)']) / comparison_df['Path Length (L)_baseline']
    ) * 100

    # Filter out the baseline itself for the plot
    plot_df_improvement = comparison_df[comparison_df['Algorithm'] != baseline_algo]

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Scenario', y='Path Length Improvement (%)', hue='Algorithm', data=plot_df_improvement, palette='plasma')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Path Length Improvement Relative to {baseline_algo}', fontsize=16, weight='bold')
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.legend(title='Algorithm', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'focused_relative_improvement_path_length.png'), dpi=300)
    plt.close()
    print("Saved Plot 3: Relative Improvement in Path Length")

    # --- Plot 4: Generalization Analysis (Box Plot) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Algorithm', y='Path Length (L)', data=df, palette='coolwarm')
    plt.title('Path Length Distribution Across All Scenarios (Generalization)', fontsize=16, weight='bold')
    plt.ylabel('Path Length (L)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'focused_generalization_boxplot.png'), dpi=300)
    plt.close()
    print("Saved Plot 4: Generalization Analysis (Box Plot)")


if __name__ == '__main__':
    DATA_FILE = 'showcases/Simulating_a_data_collection_scenario_static/raw_experimental_data_batch.csv'
    OUTPUT_DIR = 'showcases/Simulating_a_data_collection_scenario_static/plots'
    create_focused_visuals(DATA_FILE, output_dir=OUTPUT_DIR)