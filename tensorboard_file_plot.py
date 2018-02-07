import numpy as np
from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    return event_acc

if __name__ == '__main__':
    plain_train = plot_tensorflow_log(sys.argv[1])
    plain_val = plot_tensorflow_log(sys.argv[2])
    ours_train = plot_tensorflow_log(sys.argv[3])
    ours_val = plot_tensorflow_log(sys.argv[4])

    # from IPython import embed; embed()

    plain_training_accuracies = plain_train.Scalars('ep3_top1')
    plain_validation_accuracies = plain_val.Scalars('ep3_top1')
    ours_training_accuracies = ours_train.Scalars('ep3_top1')
    ours_validation_accuracies = ours_val.Scalars('ep3_top1')

    x = [0]
    for summary in plain_training_accuracies:
        x.append(summary.step + 5005)
    x = np.array(x)
    y = np.ones([len(x), 4]) * 100

    for i in xrange(1, len(x)):
        y[i, 0] = 100 - plain_training_accuracies[i-1].value
        y[i, 1] = 100 - plain_validation_accuracies[i-1].value
        y[i, 2] = 100 - ours_training_accuracies[i-1].value
        y[i, 3] = 100 - ours_validation_accuracies[i-1].value

    plt.plot(x / 1e4, y[:,0],  linestyle = ':', c = 'b' )
    plt.plot(x / 1e4, y[:,1], label='plain', c = 'b')
    plt.plot(x / 1e4, y[:,2], linestyle = ':', c = 'r')
    plt.plot(x / 1e4, y[:,3], label='ours', c = 'r')

    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel("Steps (1e4)")
    plt.ylabel("Error (%)")
    plt.title("Training Error")
    plt.legend(loc='upper right', frameon=True)
    plt.show()
