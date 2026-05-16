import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./wildtime_exp3/results/exp3_huffpost_results.csv")

plt.figure(figsize=(6,4))
plt.plot(df["time_gap"], df["accuracy"], marker="o")
plt.xlabel("Time Gap")
plt.ylabel("Accuracy")
plt.title("HuffPost Accuracy vs Time Gap")
plt.grid(True)
plt.tight_layout()
plt.savefig("./wildtime_exp3/results/huffpost_accuracy_vs_gap.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(df["time_gap"], df["ece"], marker="o")
plt.xlabel("Time Gap")
plt.ylabel("ECE")
plt.title("HuffPost ECE vs Time Gap")
plt.grid(True)
plt.tight_layout()
plt.savefig("./wildtime_exp3/results/huffpost_ece_vs_gap.png")
plt.close()

print("Saved plots.")
