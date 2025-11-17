import matplotlib.pyplot as plt

# Data
dropouts = [0.2, 0.4, 0.6, 0.8]
avg_std = [31.2098, 49.0633, 70.0010, 110.7466]

plt.figure(figsize=(8, 5))

# Blue bars
bars = plt.bar(dropouts, avg_std)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f"{height:.2f}",
             ha='center',
             va='bottom')

plt.xlabel("Dropout Rate")
plt.ylabel("Average Std Deviation")
plt.title("Dropout vs Avg Std Deviation")
plt.tight_layout()
plt.show()
