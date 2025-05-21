import matplotlib.pyplot as plt

metrics = {
    "Euclidean": {"eer": [38.95, 42.25, 43.27], "f1": [60.92, 57.55, 56.73]},
    "Normalized Euclidean": {"eer": [46.13, 47.68, 48.81], "f1": [53.87, 52.32, 51.17]},
    "L1": {"eer": [38.07, 41.28, 42.84], "f1": [61.96, 58.72, 57.16]},
    "L1 with filter": {"eer": [37.39, 41.29, 42.71], "f1": [62.59, 58.71, 57.29]},
    "L1-scaled": {"eer": [43.67, 45.00, 45.97], "f1": [56.33, 55.00, 54.02]},
    "ITAD": {"eer": [60.88, 57.77, 54.25], "f1": [39.11, 42.22, 45.77]}
}

ngrams = [1, 2, 3]

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]

best_eer = float('inf')
best_point_eer = None
worst_eer = float('-inf')
worst_point_eer = None

best_f1 = float('-inf')
best_point_f1 = None
worst_f1 = float('inf')
worst_point_f1 = None

for (name, values), color in zip(metrics.items(), colors):
    for n, eer in zip(ngrams, values["eer"]):
        if eer < best_eer:
            best_eer = eer
            best_point_eer = (n, eer, name)
        if eer > worst_eer:
            worst_eer = eer
            worst_point_eer = (n, eer, name)

    for n, f1 in zip(ngrams, values["f1"]):
        if f1 > best_f1:
            best_f1 = f1
            best_point_f1 = (n, f1, name)
        if f1 < worst_f1:
            worst_f1 = f1
            worst_point_f1 = (n, f1, name)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
fig.subplots_adjust(wspace=0.2)

ax = axs[0]
for (name, values), color in zip(metrics.items(), colors):
    ax.plot(ngrams, values["eer"], label=name, marker='o', color=color)

ax.scatter(best_point_eer[0], best_point_eer[1], color='green', zorder=5, edgecolor='black')
ax.scatter(worst_point_eer[0], worst_point_eer[1], color='red', zorder=5, edgecolor='black')

ax.annotate(f"{best_point_eer[2]}, N={best_point_eer[0]} — {best_eer:.2f}%",
             xy=(best_point_eer[0], best_point_eer[1]),
             xytext=(50, 0), textcoords='offset points',
             color='green', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='green'))

ax.annotate(f"{worst_point_eer[2]}, N={worst_point_eer[0]} — {worst_eer:.2f}%",
             xy=(worst_point_eer[0], worst_point_eer[1]),
             xytext=(100, -20), textcoords='offset points',
             color='red', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='red'))

ax.set_title("EER (%) by N-gram and Distance Metric")
ax.set_xlabel("N-gram")
ax.set_ylabel("EER (%)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xticks(ngrams)

ax = axs[1]
for (name, values), color in zip(metrics.items(), colors):
    ax.plot(ngrams, values["f1"], label=name, marker='o', color=color)

ax.scatter(best_point_f1[0], best_point_f1[1], color='green', zorder=5, edgecolor='black', label='Best Result')
ax.scatter(worst_point_f1[0], worst_point_f1[1], color='red', zorder=5, edgecolor='black', label='Worst Result')

ax.annotate(f"{best_point_f1[2]}, N={best_point_f1[0]} — {best_f1:.2f}%",
             xy=(best_point_f1[0], best_point_f1[1]),
             xytext=(100, -10), textcoords='offset points',
             color='green', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='green'))

ax.annotate(f"{worst_point_f1[2]}, N={worst_point_f1[0]} — {worst_f1:.2f}%",
             xy=(worst_point_f1[0], worst_point_f1[1]),
             xytext=(0, 20), textcoords='offset points',
             color='red', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='red'))

ax.set_title("F1 Score (%) by N-gram and Distance Metric")
ax.set_xlabel("N-gram")
ax.set_ylabel("F1 Score (%)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xticks(ngrams)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, title="Metric", bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("combined_small_plot.png")

plt.figure(figsize=(12, 5))
for (name, values), color in zip(metrics.items(), colors):
    plt.plot(ngrams, values["eer"], label=name, marker='o', color=color)

plt.scatter(best_point_eer[0], best_point_eer[1], color='green', zorder=5, edgecolor='black')
plt.scatter(worst_point_eer[0], worst_point_eer[1], color='red', zorder=5, edgecolor='black')

plt.annotate(f"{best_point_eer[2]}, N={best_point_eer[0]} — {best_eer:.2f}%",
             xy=(best_point_eer[0], best_point_eer[1]),
             xytext=(-20, 20), textcoords='offset points',
             color='green', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='green'))

plt.annotate(f"{worst_point_eer[2]}, N={worst_point_eer[0]} — {worst_eer:.2f}%",
             xy=(worst_point_eer[0], worst_point_eer[1]),
             xytext=(-20, -20), textcoords='offset points',
             color='red', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='red'))

plt.title("EER (%) by N-gram and Distance Metric")
plt.xlabel("N-gram")
plt.ylabel("EER (%)")
plt.xticks(ngrams)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("eer_plot_final_small.png")

plt.figure(figsize=(12, 5))
for (name, values), color in zip(metrics.items(), colors):
    plt.plot(ngrams, values["f1"], label=name, marker='o', color=color)

plt.scatter(best_point_f1[0], best_point_f1[1], color='green', zorder=5, edgecolor='black')
plt.scatter(worst_point_f1[0], worst_point_f1[1], color='red', zorder=5, edgecolor='black')

plt.annotate(f"{best_point_f1[2]}, N={best_point_f1[0]} — {best_f1:.2f}%",
             xy=(best_point_f1[0], best_point_f1[1]),
             xytext=(-20, 20), textcoords='offset points',
             color='green', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='green'))

plt.annotate(f"{worst_point_f1[2]}, N={worst_point_f1[0]} — {worst_f1:.2f}%",
             xy=(worst_point_f1[0], worst_point_f1[1]),
             xytext=(-20, -20), textcoords='offset points',
             color='red', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='red'))

plt.title("F1 Score (%) by N-gram and Distance Metric")
plt.xlabel("N-gram")
plt.ylabel("F1 Score (%)")
plt.xticks(ngrams)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("f1_plot_final_small.png")
