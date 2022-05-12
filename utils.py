import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skimage
from skimage import color
from skimage.util import crop


def plot(data, name, scenario, day, clock, obstacle_count=0, stats_goal=None, test=False, load_scenario=None, pretrained=False):
    title = str(stats_goal) if name == 'dfp' else 'DQN agent' if name == 'dqn' else 'A2C agent'
    title_addon = f'_obs={obstacle_count}' if scenario in [0, 4] else ''
    if pretrained: title_addon += '_pretrained'
    folder = "DFP" if name == "dfp" else "DQN" if name == "dqn" else "A2C"

    if test: folder += '/Test'

    fig, (ax1, ax2) = plt.subplots(2, figsize=(6.4, 4.8 * 2), tight_layout=True)
    fig.suptitle(title)

    data['Reward means'] = data['Rewards'].rolling(20, min_periods=1).mean()
    y = 'Episodes' if test else 'Time steps'
    data.plot(y, ['Rewards', 'Reward means'], ax=ax1)

    data['Tile means'] = data['Tiles'].rolling(20, min_periods=1).mean()
    data.plot(y, ['Tiles', 'Tile means'], ax=ax2)

    for ax in (ax1, ax2): ax.label_outer()

    # individual plots
    if test:
        fig.savefig(f'{folder}/agent_{load_scenario}/scenario_{scenario}/{day}/{name}_plot_{clock}' + title_addon + '.png')
        name_prefix = f'{folder}/agent_{load_scenario}/scenario_{scenario}/{day}/individuals/'
    else:
        fig.savefig(f'{folder}/scenario_{scenario}/statistics/{day}/{name}_plot_{clock}' + title_addon + '.png')
        name_prefix = f'{folder}/scenario_{scenario}/statistics/{day}/individuals/'

    fig, ax = plt.subplots()
    data.plot(y, ['Rewards', 'Reward means'], ax=ax, title=title)
    fig.savefig(f'{name_prefix}{name}_plot_{clock}' + title_addon + '.png')

    fig, ax = plt.subplots()
    data.plot(y, ['Tiles', 'Tile means'], ax=ax, title=title)
    fig.savefig(f'{name_prefix}{name}_plot_{clock}' + title_addon + '_tile.png')

    plt.close('all')

    if test:
        data.to_csv(f'{folder}/agent_{load_scenario}/scenario_{scenario}/{day}/{name}_stats_{clock}' + title_addon + '.csv', index_label='Episode')
    else:
        data.to_csv(f'{folder}/scenario_{scenario}/statistics/{day}/{name}_stats_{clock}' + title_addon + '.csv', index_label='Episode')


def preprocessImg(img):
    img = crop(img, ((0, 12), (6, 6), (0, 0)), copy=False)
    img = skimage.color.rgb2gray(img)
    return img
