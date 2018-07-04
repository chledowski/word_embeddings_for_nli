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


def draw_multiple_learning_curves(dirpaths, savepath='./results/lr/warmupupdown_0.5_K_1_v2.png', titles=[]):

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



if __name__ == "__main__":
    # draw_learning_curves('results/2018_03_13_15209810700_lexvec')

    draw_multiple_learning_curves(['results/lr/warmup/bn/upstill/0.5/1', 'results/lr/warmup/bn/upstill/0.5/1.5',
                                   'results/lr/warmup/bn/upstill/0.5/2', 'results/lr/0.5'],
                                  titles = ['upstill_1', 'upstill_1.5', 'upstill_2','baseline'])

