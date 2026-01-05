import csv
import matplotlib.pyplot as plt
import os
import numpy as np

# Data Loading
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def parse_model_size(model_name):
    # Extract size from "Qwen/Qwen2.5-0.5B-Instruct" -> 0.5
    if "0.5B" in model_name: return 0.5
    if "1.5B" in model_name: return 1.5
    if "3B" in model_name: return 3.0
    if "7B" in model_name: return 7.0
    return 0.0

def analyze():
    data = load_data('eval/eval_results.csv')
    
    # Organize data
    # structure: metrics[metric][mode] = {size: value, ...}
    metrics_map = {
        "Accuracy": "Accuracy",
        "Structure Violation Rate": "Structure Violation Rate",
        "Avg Total Tokens": "Avg Total Tokens"
    }
    
    plot_data = {m: {"xml": {}, "natural": {}, "glyph": {}} for m in metrics_map}
    
    for row in data:
        mode = row["Mode"]
        size = parse_model_size(row["Model"])
        
        for metric in metrics_map:
            val = float(row[metric])
            plot_data[metric][mode][size] = val

    # Sorted sizes for X-axis
    sizes = sorted([0.5, 1.5, 3.0, 7.0])
    
    output_dir = "eval/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Plots
    for metric_name, modes_dict in plot_data.items():
        plt.figure(figsize=(10, 6))
        
        for mode, size_dict in modes_dict.items():
            # Get values in order of sizes
            y_values = [size_dict.get(s, 0) for s in sizes]
            
            # Styling
            marker = 'o'
            linestyle = '-'
            if mode == 'glyph': 
                marker = 's'
                linestyle = '-'
                linewidth = 2.5
            else:
                linewidth = 1.5
                
            plt.plot(sizes, y_values, marker=marker, linestyle=linestyle, linewidth=linewidth, label=mode)
        
        plt.title(f"{metric_name} vs Model Size (Qwen 2.5 Instruct)")
        plt.xlabel("Model Size (Billions of Parameters)")
        plt.ylabel(metric_name)
        plt.xticks(sizes, [f"{s}B" for s in sizes])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        filename = f"{metric_name.lower().replace(' ', '_')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close()

    # Textual Analysis
    print("\n=== Analysis Summary ===")
    
    # 1. Best Performing Accuracy
    # Check 7B model
    best_acc = 0
    best_mode = ""
    for row in data:
        if parse_model_size(row["Model"]) == 7.0:
            acc = float(row["Accuracy"])
            if acc > best_acc:
                best_acc = acc
                best_mode = row["Mode"]
    print(f"Top Accuracy (7B): {best_acc} ({best_mode})")
    
    # 2. Token Efficiency
    # Compare avg tokens at 7B
    print("\nToken Usage (7B):")
    for row in data:
         if parse_model_size(row["Model"]) == 7.0:
             print(f"  {row['Mode']}: {row['Avg Total Tokens']} tokens")

if __name__ == "__main__":
    analyze()
