# Extract the relevant columns for plotting
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mean_accuracies_summary.csv")

top_n = [1, 2, 3]
mood = df.iloc[0][['Mood_top_1_accuracy', 'Mood_top_2_accuracy', 'Mood_top_3_accuracy']].values
anxiety = df.iloc[0][['Anxiety_top_1_accuracy', 'Anxiety_top_2_accuracy', 'Anxiety_top_3_accuracy']].values
stress = df.iloc[0][['Stress_top_1_accuracy', 'Stress_top_2_accuracy', 'Stress_top_3_accuracy']].values

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(top_n, mood, marker='o', linestyle='--', label='Mood')
plt.plot(top_n, anxiety, marker='s', linestyle='--', label='Anxiety')
plt.plot(top_n, stress, marker='^', linestyle='--', label='Stress')

plt.xlabel('Top-n')
plt.ylabel('Accuracy (%)')
plt.title('Top-n Accuracy for Mood, Anxiety, and Stress')
plt.xticks(top_n)
# plt.ylim(0, 100)
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
