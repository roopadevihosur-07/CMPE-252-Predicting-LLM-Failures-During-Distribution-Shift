import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./wildtime_exp3/results/exp3_arxiv_results.csv")

plt.figure(figsize=(6, 4))
plt.plot(df["time_gap"], df["accuracy"], marker="o")
plt.xlabel("Time Gap")
plt.ylabel("Accuracy")
plt.title("ArXiv Accuracy vs Time Gap")
plt.grid(True)
plt.tight_layout()
plt.savefig("./wildtime_exp3/results/arxiv_accuracy_vs_gap.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(df["time_gap"], df["ece"], marker="o")
plt.xlabel("Time Gap")
plt.ylabel("ECE")
plt.title("ArXiv ECE vs Time Gap")
plt.grid(True)
plt.tight_layout()
plt.savefig("./wildtime_exp3/results/arxiv_ece_vs_gap.png")
plt.close()

print("Saved plots.")
