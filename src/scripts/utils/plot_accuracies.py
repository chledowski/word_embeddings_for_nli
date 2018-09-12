import pandas as pd
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_learning_curves(dirpath, savepath=None):

    history = pd.read_csv(os.path.join(dirpath, 'history.csv'))
    print(history['acc'])

    f, axes = plt.subplots(2, 2, sharex='all', sharey='col')
    axes[0,0].plot(history['acc'], label='train acc')
    axes[0,0].set_title('train acc')
    axes[0,1].plot(history['loss'], label='train loss')
    axes[0,1].set_title('train loss')
    axes[1,0].plot(history['val_acc'], label='val acc')
    axes[1,0].set_title('val acc')
    axes[1,1].plot(history['val_loss'], label='val loss')
    axes[1,1].set_title('val loss')

    if savepath:
        f.savefig(savepath)
    else:
        f.savefig(os.path.join(".", os.path.join(dirpath, 'learning_curves.png')))


def draw_multiple_learning_curves(dirpaths, savepath='./results/plot1.png', titles=[]):

    l = len(dirpaths)
    f, axes = plt.subplots(2, 2)
    axes[0, 0].set_title('train acc')
    axes[0, 1].set_title('train loss')
    axes[1, 0].set_title('val acc')
    axes[1, 1].set_title('val loss')

    for i in range(l):
        history = pd.read_csv(os.path.join(dirpaths[i], 'history.csv'))

        axes[0, 0].plot(history['acc'], label=titles[i])
        axes[0, 1].plot(history['loss'], label=titles[i])
        axes[1, 0].plot(history['val_acc'], label=titles[i])
        axes[1, 1].plot(history['val_loss'], label=titles[i])

    axes[0,0].legend()
    axes[0,1].legend()
    axes[1,0].legend()
    axes[1,1].legend()

    plt.legend()

    if not os.path.exists("./results/plots/"):
        os.makedirs("./results/plots/")

    f.savefig(savepath)

def draw_easy_hard_vs_acc(dirpaths, savepath='./results/plot1.png', titles=[]):

    l = len(dirpaths)
    f, axes = plt.subplots(2, sharex='all', sharey='all')
    axes[0].set_title('acc vs easy_acc')
    axes[1].set_title('acc vs hard_acc')

    for j in range(l):
        print(j)
        history = pd.read_csv(os.path.join(dirpaths[j], 'history.csv'))
        easy_hard = pd.read_csv(os.path.join(dirpaths[j], 'easy_hard_dataset_acc.csv'))
        print (easy_hard[['easy']])
        # for i in range(len(easy_hard[['easy']])):
        #     print(easy_hard[['easy']][i])
        #     if isinstance(easy_hard[['easy']][i], str):
        #         easy_hard[['easy']][i] = eval(easy_hard[['easy']][i])
        #     easy_hard[['easy']][i] = round(easy_hard[['easy']][i], 3)
        #     easy_hard[['hard']][i] = eval(easy_hard[['hard']][i])
        #     if isinstance(easy_hard[['hard']][i], list):
        #         easy_hard[['hard']][i] = easy_hard[['hard']][i][0]
        #     easy_hard[['hard']][i] = round(easy_hard[['hard']][i], 3)
        # print(easy_hard)

        axes[0].plot(history['acc'], easy_hard[['easy']], label=titles[j])
        axes[1].plot(history['acc'], easy_hard[['hard']], label=titles[j])

    axes[0].legend()
    axes[1].legend()

    plt.legend()

    if not os.path.exists("./results/plots/"):
        os.makedirs("./results/plots/")

    f.savefig(savepath)

if __name__ == "__main__":
    # draw_learning_curves('results/2018_03_13_15209810700_lexvec')

    draw_easy_hard_vs_acc(['results/eval_snli_lr_0.01', 'results/eval_snli_lr_0.004', 'results/eval_snli_lr_0.002', 'results/eval_snli_lr_0.001', 'results/eval_snli_lr_0.0004', ],
                                  titles = ['0.01', '0.004', '0.002','0.001', '0.0004'])

