import matplotlib.pyplot as plt
import numpy as np

def show_image_examples(df_iterator):
    all_labels = list(df_iterator.class_indices.keys())
    iter_image, iter_labels = next(df_iterator)
    fig, ax = plt.subplots(2, 4)
    for image, labels, plot in zip(iter_image, iter_labels, ax.flatten()):
        plot.imshow(image[:, :, 0], cmap="bone")
        plot.set_title(
            ", ".join([class_name for class_name, belongs in zip(all_labels, labels) if belongs == 1])
        )
        plot.axis("off")

    plt.show()

def plot_class_count(df, labels):
    label_counts = df['Finding Labels'].value_counts()[1:len(labels)]
    
    fig, ax1 = plt.subplots(1,1)
    bars = ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
    ax1.bar_label(bars)

    ax1.set_xticks(np.arange(len(label_counts))+0.5)
    _ = ax1.set_xticklabels(label_counts.index, rotation = 90)

    plt.show()