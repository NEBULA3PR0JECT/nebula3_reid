import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# import torch
# import torch.nn as nn
# from sklearn.metrics import accuracy_score
# import time
# import pickle
import os


# from sklearn.datasets import load_digits

def umap_plot(labeled_embed, result_path, metric='cosine'):
    # digits = load_digits()
    # metric = 'correlation'  # "euclidean"
    min_dist = 0.1 # This controls how tightly the embedding is allowed compress points together. Larger values ensure embedded points are more evenly distributed, while smaller values allow the algorithm to optimise more accurately with regard to local structure. Sensible values are in the range 0.001 to 0.5, with 0.1 being a reasonable default.
    n_neighbors = 5 #This determines the number of neighboring points used in local approximations of manifold structure. Larger values will result in more global structure being preserved at the loss of detailed local structure. In general this parameter should often be in the range 5 to 50, with a choice of 10 to 15 being a sensible default.
    import umap
    import umap.plot
    embed_mat = np.stack(labeled_embed.embed)
    labels = np.array(labeled_embed.label)

    lbl, c = np.unique(labeled_embed.label, return_counts=True)
    rmv_labels = np.unique(lbl[c<n_neighbors])
    ind_rmv = np.concatenate([np.where(labeled_embed.label == rmv_labe)[0] for rmv_labe in rmv_labels])
    embed_mat = np.delete(embed_mat, ind_rmv, axis=0)
    labels = np.delete(labels, ind_rmv, axis=0)

    mapper = umap.UMAP(n_neighbors=n_neighbors,
                      min_dist=min_dist,
                      metric=metric).fit(embed_mat)
    ax = umap.plot.points(mapper, labels=labels)
    ax.figure.savefig(os.path.join(result_path, 'metric_' + str(metric) + '_n_neighbors_' + str(n_neighbors) + '_min_dist_' + str(min_dist) + '_umap.pdf'))

def roc_plot(labels, predictions, positive_label, save_dir, thresholds_every=5, unique_id=''):
    # roc_auc_score assumes positive label is 0 namely FINGER_IDX=0 or equivalently positive_label = 1
    # os.environ['DISPLAY'] = str('localhost:10.0')
    # qt bug ???
    os.environ['QT_XKB_CONFIG_ROOT'] = '/usr/share/X11/xkb/'
    assert positive_label == 1

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions,
                                            pos_label=positive_label)

    auc = sklearn.metrics.roc_auc_score(labels, predictions) # TODO consider replace with metrics.auc(fpr, tpr) since it has the label built in implicit
    print("AUC: {}".format(auc))
    granularity_percentage = 1. / labels.shape[0] *100
    lw = 2
    n_labels = len(labels)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f) support = %3d' % (auc, n_labels))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC model {} (gran={:.2e}[%])".format(unique_id, granularity_percentage))
    plt.legend(loc="lower right")

    # plot some thresholds
    thresholdsLength = len(thresholds) #- 1
    thresholds_every = int(thresholdsLength/thresholds_every)
    thresholds = thresholds[1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

    colorMap = plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
        plt.text(fpr[i] - 0.03, tpr[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                 color=colorMap(i / thresholdsLength))

    filename = unique_id + 'roc_curve.png'
    plt.savefig(os.path.join(save_dir, filename), format="png")


def p_r_plot_multi_class(all_targets, all_predictions, save_dir, thresholds_every_in=5, unique_id=None, classes=[0, 1, 2]):
    # Precision recall  assumes positive label is 0 namely FINGER_IDX=0 or equivalently positive_label = 1
    # os.environ['DISPLAY'] = str('localhost:10.0')
    # qt bug ???
    os.environ['QT_XKB_CONFIG_ROOT'] = '/usr/share/X11/xkb/'

    all_targets_one_hot = label_binarize(all_targets, classes=classes)
    precision = dict()
    recall = dict()
    thresholds_ap = dict()
    average_precision = dict()
    for i in range(all_predictions.shape[1]):
        precision[i], recall[i], thresholds_ap[i] = precision_recall_curve(all_targets_one_hot[:, i],
                                                            all_predictions[:, i])
        average_precision[i] = average_precision_score(all_targets_one_hot[:, i], all_predictions[:, i])

        granularity_percentage = 1. / all_targets.shape[0] *100
        lw = 2
        n_labels = all_targets.shape[0]

        plt.figure()
        plt.plot(recall[i], precision[i], color='darkorange',
                 lw=lw, label='AP (area = %0.3f) support = %3d' % (average_precision[i], n_labels))
        plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("PR (Macro) 1vs.all :{} model {} (gran={:.2e}[%])".format(i, unique_id, granularity_percentage))
        plt.legend(loc="lower right")

        # plot some thresholds
        thresholdsLength = len(thresholds_ap[i]) #- 1
        thresholds_every = max(int(thresholdsLength/thresholds_every_in), 1)

        thresholds = thresholds_ap[i][1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
        thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

        colorMap = plt.get_cmap('jet', thresholdsLength)
        precision_cls = precision[i]
        recall_cls = recall[i]
        for ind in range(0, thresholdsLength, thresholds_every):
            threshold_value_with_max_four_decimals = thresholds[ind].__format__('.3f')
            plt.text(recall_cls[ind] - 0.03, precision_cls[ind] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                     color=colorMap(ind / thresholdsLength))

        filename = unique_id + 'p_r_curve_class_' + str(i) + '.png'
        plt.savefig(os.path.join(save_dir, filename), format="png")

    if 0:
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], thresholds_micro = precision_recall_curve(all_targets_one_hot.ravel(),
                                                                        all_predictions.ravel())

        average_precision["micro"] = average_precision_score(all_targets_one_hot, all_predictions,
                                                             average="micro")

        plt.step(recall['micro'], precision['micro'], where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))

        # plot some thresholds
        thresholdsLength = len(thresholds_micro)  # - 1
        thresholds_every = max(int(thresholdsLength / thresholds_every_in), 1)

        thresholds = thresholds_micro[1:]  # `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
        thresholdsLength = thresholdsLength - 1  # changed the threshold vector henc echange the len

        colorMap = plt.get_cmap('jet', thresholdsLength)
        for ind in range(0, thresholdsLength, thresholds_every):
            threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
            plt.text(recall["micro"][ind] - 0.03, precision["micro"][ind] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                     color=colorMap(ind / thresholdsLength))

        filename = unique_id + 'p_r_micro_curve.png'
        plt.savefig(os.path.join(save_dir, filename), format="png")



def p_r_plot(labels, predictions, positive_label, save_dir, thresholds_every=5, unique_id=None):
    # Precision recall  assumes positive label is 0 namely FINGER_IDX=0 or equivalently positive_label = 1
    # os.environ['DISPLAY'] = str('localhost:10.0')
    # qt bug ???
    os.environ['QT_XKB_CONFIG_ROOT'] = '/usr/share/X11/xkb/'
    assert positive_label == 1

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, predictions,
                                                                pos_label=positive_label)

    ap = sklearn.metrics.average_precision_score(labels, predictions,
                                                                pos_label=positive_label)

    print("AP : {}".format(ap))
    # auc = sklearn.metrics.roc_auc_score(labels, predictions)
    granularity_percentage = 1. / labels.shape[0] *100
    lw = 2
    n_labels = len(labels)

    plt.figure()
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='AP (area = %0.3f) support = %3d' % (ap, n_labels))
    plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR model {} (gran={:.2e}[%])".format(unique_id, granularity_percentage))
    plt.legend(loc="lower right")

    # plot some thresholds
    thresholdsLength = len(thresholds) #- 1
    thresholds_every = max(int(thresholdsLength/thresholds_every), 1)

    thresholds = thresholds[1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

    colorMap = plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
        plt.text(recall[i] - 0.03, precision[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                 color=colorMap(i / thresholdsLength))

    filename = unique_id + 'p_r_curve.png'
    plt.savefig(os.path.join(save_dir, filename), format="png")


def plot_tsne(all_targets, all_features, path):
    from sklearn.manifold import TSNE
    n_components = 2
    perplexity = 30
    tsne_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(all_features)
    tsne_data.shape
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=all_targets, s=1)
    plt.title("TSNE of classes {} N={} p={}".format(np.unique(all_targets), all_targets.shape[0], perplexity))
    plt.savefig(os.path.join(path, 'tsne_' 'p_' + str(perplexity) + '.png'))

