from calendar import EPOCH
import matplotlib.pyplot as plt
from consts import EPOCHS, PLOTS_PATH, RESULTS_PATH, FOLDS
import numpy as np
import json

def add_last_value(ax, data, key):
    ax.annotate('%0.2f' % data[key][EPOCHS-1], xy=(1, data[key][EPOCHS-1]), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

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
    plt.clf()

def plot_class_count(title, df, labels):
    label_counts = df['Finding Labels'].value_counts()[1:len(labels)]
    
    fig, ax1 = plt.subplots(1,1)
    fig.suptitle(title, fontsize=16)
    bars = ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
    ax1.bar_label(bars)

    ax1.set_xticks(np.arange(len(label_counts))+0.5)
    _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
    
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + f'{title.lower().replace(" ", "_")}_count.png')
    plt.show()
    plt.clf()

def plot_metrics(title, model_history):

    fig, axs = plt.subplots(2)
    suptitle = fig.suptitle(title.replace('_', ' ').title(), fontsize=16)

    axs[0].plot(np.arange(0, EPOCHS), model_history["accuracy"], label="Accuracy")
    axs[0].set_title("Accuracy")
    add_last_value(axs[0], model_history, "accuracy")

    axs[1].plot(np.arange(0, EPOCHS), model_history["loss"], label="Loss")
    axs[1].set_title("Loss ")
    add_last_value(axs[1], model_history, "loss")

    legend = fig.legend(loc='center right', bbox_to_anchor = (1.15, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_PATH + f'{title}_plots.png', bbox_extra_artists=(legend, suptitle), bbox_inches='tight')
    plt.show()
    plt.cla()
    

def plot_augmentation(results):
    no_augmentation = results[0]
    augmentation = results[1]

    fig, axs = plt.subplots(2)
    suptitle = fig.suptitle('Usage of Augmentation results', fontsize=16)

    axs[0].plot(np.arange(0, EPOCHS), no_augmentation["accuracy"], label=f"No Augmentation Accuracy ({no_augmentation['accuracy'][EPOCHS -1]:.2f})")
    axs[0].plot(np.arange(0, EPOCHS), augmentation["accuracy"], label=f"Augmentation  Accuracy ({augmentation['accuracy'][EPOCHS -1]:.2f})")
    axs[0].set_title(f"Accuracy")
    legend1 = axs[0].legend(loc='upper right', bbox_to_anchor = (1.6, 0.5))
    
    axs[1].plot(np.arange(0, EPOCHS), no_augmentation["loss"], label=f"No Augmentation  Loss ({no_augmentation['loss'][EPOCHS -1]:.2f})")
    axs[1].plot(np.arange(0, EPOCHS), augmentation["val_loss"], label=f"Augmentation  Loss ({augmentation['loss'][EPOCHS -1]:.2f})")
    axs[1].set_title(f"Loss")
    legend2 = axs[1].legend(loc='upper right', bbox_to_anchor = (1.6, 0.5))

    # fig.tight_layout()
    fig.savefig(PLOTS_PATH + f'aug_compare_plot.png', bbox_extra_artists=(legend1, legend2, suptitle), bbox_inches='tight')
    plt.show()
    plt.cla()

def plot_activations_funcs(results):
    fig, axs = plt.subplots(2)
    suptitle = fig.suptitle('Usage of different Activation Functions', fontsize=16)

    axs[0].plot(np.arange(0, EPOCHS), results[0]["accuracy"], label=f'Relu Accuracy {results[0]["accuracy"][EPOCHS-1]:.2f}')
    axs[0].plot(np.arange(0, EPOCHS), results[1]["accuracy"], label=f'Sigmoid  Accuracy {results[1]["accuracy"][EPOCHS-1]:.2f}')
    axs[0].plot(np.arange(0, EPOCHS), results[2]["accuracy"], label=f'Tanh  Accuracy {results[2]["accuracy"][EPOCHS-1]:.2f}')
    axs[0].set_title("Accuracy")
    legend1 = axs[0].legend(loc='center right', bbox_to_anchor = (1.5, 0.5))
    
    axs[1].plot(np.arange(0, EPOCHS), results[0]["loss"], label=f'Relu Loss ({results[0]["loss"][EPOCHS-1]:.2f})')
    axs[1].plot(np.arange(0, EPOCHS), results[1]["loss"], label=f'Sigmoid Loss ({results[1]["loss"][EPOCHS-1]:.2f})')
    axs[1].plot(np.arange(0, EPOCHS), results[2]["loss"], label=f'Tanh Loss ({results[2]["loss"][EPOCHS-1]:.2f})')
    axs[1].set_title("Loss")
    legend2 = axs[1].legend(loc='center right', bbox_to_anchor = (1.45, 0.5))
    
    # fig.tight_layout()
    fig.savefig(PLOTS_PATH + f'activation_func_compare_plot.png', bbox_extra_artists=(legend1, legend2, suptitle), bbox_inches='tight')
    plt.show()
    plt.cla()

def plot_kernels(results):
    fig, axs = plt.subplots(2)
    suptitle = fig.suptitle('Usage of different kernels', fontsize=16)

    axs[0].plot(np.arange(0, EPOCHS), results[0]["accuracy"], label=f"3x3 Accuracy ({results[0]['accuracy'][EPOCHS-1]:.2f})")
    axs[0].plot(np.arange(0, EPOCHS), results[1]["accuracy"], label=f"5x5  Accuracy ({results[1]['accuracy'][EPOCHS-1]:.2f})")
    axs[0].plot(np.arange(0, EPOCHS), results[2]["accuracy"], label=f"7x7  Accuracy ({results[2]['accuracy'][EPOCHS-1]:.2f})")
    axs[0].plot(np.arange(0, EPOCHS), results[3]["accuracy"], label=f"12x12  Accuracy ({results[3]['accuracy'][EPOCHS-1]:.2f})")
    axs[0].set_title("Accuracy")
    legend1 = axs[0].legend(loc='center right', bbox_to_anchor = (1.5, 0.5))
    
    axs[1].plot(np.arange(0, EPOCHS), results[0]["loss"], label=f"3x3 Loss ({results[0]['loss'][EPOCHS-1]:.2f})")
    axs[1].plot(np.arange(0, EPOCHS), results[1]["loss"], label=f"5x5 Loss ({results[1]['loss'][EPOCHS-1]:.2f})")
    axs[1].plot(np.arange(0, EPOCHS), results[2]["loss"], label=f"7x7 Loss ({results[2]['loss'][EPOCHS-1]:.2f})")
    axs[1].plot(np.arange(0, EPOCHS), results[3]["loss"], label=f"12x12 Loss ({results[3]['loss'][EPOCHS-1]:.2f})")
    axs[1].set_title("Loss")
    legend2 = axs[1].legend(loc='center right', bbox_to_anchor = (1.4, 0.5))
    
    # fig.tight_layout()
    fig.savefig(PLOTS_PATH + f'kernels.png', bbox_extra_artists=(legend1, legend2, suptitle), bbox_inches='tight')
    plt.show()
    plt.cla()


def cmp_avgs(files, keys, title):
    for file, key in zip(files, keys):
        with open(RESULTS_PATH + file, 'r') as res_file:
            results = json.load(res_file)[key]
            results_matrix = np.zeros((FOLDS, EPOCHS))
            for fold in results:
                fold_results = fold['history']['accuracy']
                results_matrix[fold["fold"] -1] = fold_results
            
            avg = np.average(results_matrix, axis=0)
            plt.plot(np.arange(0, EPOCHS), avg, label=f'{key} folds avg')
    plt.title = title.title()
    plt.legend()
    plt.savefig(PLOTS_PATH + title.replace(' ', '_')+'.png')
    plt.cla()
    plt.clf()
    plt.show()


def plot_folds(file_name, data_key):
    METRICS = ['accuracy', 'loss']
    
    with open(RESULTS_PATH + file_name, 'r') as res_file:
        results = json.load(res_file)[data_key]
        for metric in METRICS:
            fig, axs = plt.subplots(2)
            suptitle = fig.suptitle(data_key.replace('_', ' ').title() + ' ' + metric, fontsize=16)
            results_matrix = np.zeros((FOLDS, EPOCHS))
            for fold in results:
                fold_results = fold['history'][metric]
                results_matrix[fold["fold"] -1] = fold_results
                axs[0].plot(np.arange(0, EPOCHS), fold_results, label=f'Fold {fold["fold"]}')
            axs[0].legend()

            avg = np.average(results_matrix, axis=0)
            axs[1].plot(np.arange(0, EPOCHS), avg, label=f'All folds avg')
            axs[1].legend()
            fig.savefig(PLOTS_PATH + f"{data_key}_{metric}.png")
            plt.cla()
            plt.clf()
            plt.show()

plot_folds('no_aug.json', 'no_augmentation')
plot_folds('aug.json', 'augmentation')
cmp_avgs(['no_aug.json', 'aug.json'], ['no_augmentation', 'augmentation'], "Augmentation vs No Augmentation")
