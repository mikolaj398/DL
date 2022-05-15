import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

PLOTS_PATH = './plots/'
def show_image_examples(df_iterator):
    all_labels = list(df_iterator.class_indices.keys())
    iter_image, iter_labels = next(df_iterator)
    fig, ax = plt.subplots(1, 4)
    for image, labels, plot in zip(iter_image, iter_labels, ax.flatten()):
        plot.imshow(image[:, :, 0], cmap="bone")
        plot.set_title(
            ", ".join([class_name for class_name, belongs in zip(all_labels, labels) if belongs == 1])
        )
        plot.axis("off")

    plt.savefig(PLOTS_PATH + 'images_example.png')
    plt.show()

def plot_class_count(df, labels):
    label_counts = df['Finding Labels'].value_counts()[1:len(labels)]
    
    fig, ax1 = plt.subplots(1,1)
    bars = ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
    ax1.bar_label(bars)

    ax1.set_xticks(np.arange(len(label_counts))+0.5)
    _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
    
    plt.savefig(PLOTS_PATH + 'class_count.png')
    plt.show()

def plot_roc(all_labels, pred_Y, test_Y):
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    for (idx, c_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    
    fig.savefig('roc.png')
    plt.show()

def plot_metrics(model_history):

    num_epochs = model_history.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(np.arange(0, num_epochs), model_history["accuracy"], label="Training Accuracy")
    ax1.plot(np.arange(0, num_epochs), model_history["loss"], label="Training Loss")
    ax1.set_title("Training")
    
    ax2.plot(np.arange(0, num_epochs), model_history["val_accuracy"], label="Validation  Accuracy")
    ax2.plot(np.arange(0, num_epochs), model_history["val_loss"], label="Validation  Loss")
    ax2.set_title("Validation ")
    ax2.legend()
    
    fig.savefig(PLOTS_PATH + 'metrics.png')
    plt.show()