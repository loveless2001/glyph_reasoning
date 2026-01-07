
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
csv_path = "glyph_reasoning/experiments/results/revision_retrieval_llama.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit()

df = pd.read_csv(csv_path)

# Calculate Accuracy per k
summary = df.groupby('k').agg({
    'A_acc': 'mean',
    'B_acc': 'mean',
    'C_acc': 'mean',
    'A_margin': 'mean',
    'C_margin': 'mean'
}).reset_index()

# Plotting
plt.figure(figsize=(14, 6))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(summary['k'], summary['A_acc'], marker='o', label='Base (A)', linewidth=2, color='#3498db')
plt.plot(summary['k'], summary['B_acc'], marker='s', label='Base+Glyph (B)', linewidth=2, linestyle='--', color='#95a5a6')
plt.plot(summary['k'], summary['C_acc'], marker='^', label='Glyph-SFT (C)', linewidth=2, color='#e74c3c')

plt.title('Retrieval Accuracy vs. Interference (k)')
plt.xlabel('Number of Distractor Turns (k)')
plt.ylabel('Accuracy')
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Subplot 2: Logit Margin (for k where it was computed)
# Margins were only computed for k >= 5 usually, but csv might have 0.0 for others.
# Filter k where margin != 0.0 or explicitly computed.
# In the last run, we enabled it for k >= 5.
margin_df = summary[summary['k'] >= 5]

plt.subplot(1, 2, 2)
if not margin_df.empty:
    width = 0.35
    x = margin_df['k']
    plt.bar(x - width/2, margin_df['A_margin'], width, label='Base (A)', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, margin_df['C_margin'], width, label='Glyph-SFT (C)', color='#e74c3c', alpha=0.8)
    
    plt.title('Logit Margin (Confidence) at k >= 5')
    plt.xlabel('Number of Distractor Turns (k)')
    plt.ylabel('Margin (LogProb Correct - Fake)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No Margin Data Available', ha='center', va='center')

plt.tight_layout()
output_path = "glyph_reasoning/experiments/results/revision_retrieval_plot.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
