import matplotlib.pyplot as plt

metrics = {
    "Расстояние Евклида": {"eer": [29.84, 30.82, 33.16], "f1": [70.16, 69.18, 66.84]},
    "Нормализованное расстояние Евклида": {"eer": [29.75, 30.93, 34.48], "f1": [70.25, 69.07, 65.52]},
    "L1": {"eer": [26.78, 30.04, 33.27], "f1": [73.22, 69.96, 66.73]},
    "L1 с фильтрацией": {"eer": [23.58, 26.45, 31.05], "f1": [76.42, 73.55, 68.95]},
    "L1 с масштабированием": {"eer": [18.72, 21.81, 27.10], "f1": [81.28, 78.19, 72.90]},
    "ITAD": {"eer": [19.01, 21.22, 28.51], "f1": [80.99, 78.78, 71.49]}
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
             xytext=(-330, -20), textcoords='offset points',
             color='red', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='red'))

ax.set_title("EER (%) по N-граммам и метрикам расстояния")
ax.set_xlabel("N-граммы")
ax.set_ylabel("EER (%)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xticks(ngrams)

ax = axs[1]
for (name, values), color in zip(metrics.items(), colors):
    ax.plot(ngrams, values["f1"], label=name, marker='o', color=color)

ax.scatter(best_point_f1[0], best_point_f1[1], color='green', zorder=5, edgecolor='black', label='Лучший результат')
ax.scatter(worst_point_f1[0], worst_point_f1[1], color='red', zorder=5, edgecolor='black', label='Худший результат')

ax.annotate(f"{best_point_f1[2]}, N={best_point_f1[0]} — {best_f1:.2f}%",
             xy=(best_point_f1[0], best_point_f1[1]),
             xytext=(100, -10), textcoords='offset points',
             color='green', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='green'))

ax.annotate(f"{worst_point_f1[2]}, N={worst_point_f1[0]} — {worst_f1:.2f}%",
             xy=(worst_point_f1[0], worst_point_f1[1]),
             xytext=(-330, 20), textcoords='offset points',
             color='red', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             arrowprops=dict(arrowstyle="->", color='red'))

ax.set_title("F1 (%) по N-граммам и метрикам расстояния")
ax.set_xlabel("N-граммы")
ax.set_ylabel("F1 (%)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xticks(ngrams)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, title="Metric", bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("combined_huge_plot.png")

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

plt.title("EER (%) по N-граммам и метрикам расстояния")
plt.xlabel("N-граммы")
plt.ylabel("EER (%)")
plt.xticks(ngrams)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Метрика", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("eer_plot_final_huge.png")

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

plt.title("F1 (%) по N-граммам и метрикам расстояния")
plt.xlabel("N-граммы")
plt.ylabel("F1 (%)")
plt.xticks(ngrams)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Метрика", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("f1_plot_final_huge.png")
