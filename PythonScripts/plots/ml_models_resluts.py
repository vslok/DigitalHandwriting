import matplotlib.pyplot as plt

# Данные для каждого классификатора
metrics = {
    "Машина опорных векторов (SVM)": {
        "accuracy": [97.98, 97.88, 97.92],
        "precision": [99.68, 99.63, 99.42],
        "recall": [96.27, 96.12, 96.41],
        "f1": [97.92, 97.82, 97.88]
    },
    "Наивный байесовский классификатор (Naive Bayes)": {
        "accuracy": [88.54, 84.03, 81.39],
        "precision": [87.46, 82.11, 78.78],
        "recall": [91.27, 89.65, 89.48],
        "f1": [89.04, 85.18, 83.15]
    },
    "Случайный лес (Random Forest)": {
        "accuracy": [98.57, 98.33, 97.78],
        "precision": [99.98, 99.98, 99.96],
        "recall": [97.16, 96.69, 95.61],
        "f1": [98.53, 98.29, 97.71]
    },
    "XGBoost": {
        "accuracy": [98.52, 98.56, 97.99],
        "precision": [100.00, 100.00, 99.94],
        "recall": [97.04, 97.12, 96.04],
        "f1": [98.48, 98.52, 97.93]
    },
    "K-ближайших соседей (KNN)": {
        "accuracy": [97.57, 97.00, 96.31],
        "precision": [96.02, 95.51, 95.34],
        "recall": [99.37, 98.78, 97.63],
        "f1": [97.64, 97.08, 96.41]
    },
    "Многослойный перцептрон (MLP)": {
        "accuracy": [95.68, 95.20, 94.51],
        "precision": [97.06, 96.83, 94.99],
        "recall": [94.27, 93.49, 94.08],
        "f1": [95.60, 95.07, 94.49]
    },
    "Одномерная сверточная нейронная сеть (1D CNN)": {
        "accuracy": [98.22, 98.52, 98.38],
        "precision": [99.98, 100.00, 99.84],
        "recall": [96.45, 97.04, 96.62],
        "f1": [98.17, 98.48, 98.34]
    },
    "Долгая краткосрочная память (LSTM)": {
        "accuracy": [97.38, 97.35, 96.82],
        "precision": [98.35, 98.57, 97.35],
        "recall": [96.41, 96.10, 96.31],
        "f1": [97.35, 97.31, 96.81]
    },
    "Управляемые рекуррентные блоки (GRU)": {
        "accuracy": [97.62, 97.44, 96.90],
        "precision": [98.86, 98.54, 97.53],
        "recall": [96.53, 96.31, 96.27],
        "f1": [97.58, 97.40, 96.87]
    }
}

ngrams = [1, 2, 3]

# Цвета для разных моделей
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'
]

# --- Подготовка графиков ---

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()  # Чтобы проще было итерировать

metric_names = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Меткость (accuracy) (%)', 'Точность (precision) (%)', 'Полнота (recall) (%)', 'F1 (%)']

for i, (metric_key, title) in enumerate(zip(metric_names, titles)):
    ax = axes[i]
    for (model_name, model_data), color in zip(metrics.items(), colors):
        ax.plot(ngrams, model_data[metric_key], label=model_name, marker='o', color=color)

    ax.set_title(title)
    ax.set_xlabel("N-граммы")
    ax.set_ylabel("Значение метрики (%)")
    ax.set_xticks(ngrams)
    ax.grid(True, linestyle='--', alpha=0.6)

# --- Одна общая легенда ---
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, title="Classifier", bbox_to_anchor=(0.5, -0.03))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("combined_metrics_grid.png",  bbox_inches='tight')
print("Графики сохранены как 'combined_metrics_grid.png'")
