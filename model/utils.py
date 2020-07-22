import itertools
import re
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

from prettytable import PrettyTable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += val
        self.count += n
        self.avg = float(self.sum) / self.count


class ConfusionMatrix(object):

    def __init__(self, num_classes=4):
        self.matrix = np.zeros([num_classes, num_classes])

    def update(self, tlabel, plabel, count):
        self.matrix[tlabel, plabel] += count

    def to_tabel(self):
        return matrix2pretty_table(self.matrix)


def matrix2pretty_table(matrix):
    field_names = ["Label"]
    field_names.extend("p_" + str(int(i)) for i in range(matrix.shape[0]))
    table = PrettyTable(field_names)
    for t in range(matrix.shape[0]):
        pred = ["t_" + str(t)]
        pred.extend(str(int(matrix[t][p])) for p in range(matrix.shape[0]))
        table.add_row(pred)

    table.align[0] = 'l'
    # print(table)
    return table


#
#
# import numpy as np
#
# matrix = np.arange(0,16).reshape([4,4])
# print_matrix(matrix)


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          tensor_name='Confusion matrix',
                          normalize=False):
    """
    Parameters:
        confusion_matrix : confusion_matrix, numpy array
        labels       : This is a lit of labels which will be used to display the axix labels
        tensor_name : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figsize, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.

    """

    cm = confusion_matrix.astype('int')
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    # fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')

    classes = [
        re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
        for x in labels
    ]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_xticks(tick_marks)
    # ax.set_xticklabels(classes, fontsize=24, rotation=-90, ha='center')
    ax.set_xticklabels(classes, fontsize=12, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j,
                i,
                format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                horizontalalignment="center",
                fontsize=24,
                verticalalignment='center',
                color="black")
    fig.set_tight_layout(True)
    summary = fig
    return summary
