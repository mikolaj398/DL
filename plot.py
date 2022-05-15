import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from consts import EPOCHS, PLOTS_PATH

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

def plot_metrics(title, model_history):

    fig, axs = plt.subplots(2)
    suptitle = fig.suptitle(title.replace('_', ' ').title(), fontsize=16)

    axs[0].plot(np.arange(0, EPOCHS), model_history["accuracy"], label="Training Accuracy")
    axs[0].plot(np.arange(0, EPOCHS), model_history["val_accuracy"], label="Validation  Accuracy")
    axs[0].set_title("Accuracy")
    
    axs[1].plot(np.arange(0, EPOCHS), model_history["loss"], label="Training Loss")
    axs[1].plot(np.arange(0, EPOCHS), model_history["val_loss"], label="Validation  Loss")
    axs[1].set_title("Loss ")

    legend = fig.legend(loc='center right', labels = ["Training", "Validation"], bbox_to_anchor = (1.2, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_PATH + f'{title}_plots.png', bbox_extra_artists=(legend, suptitle), bbox_inches='tight')
    plt.show()
    

def plot_augmentation(results):
    no_augmentation = results[0]
    augmentation = results[1]

    fig, axs = plt.subplots(2)
    suptitle = fig.suptitle('Usage of Augmentation results', fontsize=16)

    axs[0].plot(np.arange(0, EPOCHS), no_augmentation["accuracy"], label="No Augmentation Accuracy")
    axs[0].plot(np.arange(0, EPOCHS), augmentation["accuracy"], label="Augmentation  Accuracy")
    axs[0].set_title("Accuracy")
    
    axs[1].plot(np.arange(0, EPOCHS), no_augmentation["loss"], label="No Augmentation Loss")
    axs[1].plot(np.arange(0, EPOCHS), augmentation["val_loss"], label="Augmentation  Loss")
    axs[1].set_title("Loss ")
    
    legend = fig.legend(loc='center right', labels = ["No Augmentation", "Augmentation"], bbox_to_anchor = (1.4, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_PATH + f'aug_compare_plot.png', bbox_extra_artists=(legend, suptitle), bbox_inches='tight')
    plt.show()

