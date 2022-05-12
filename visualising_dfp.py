import numpy as np
from matplotlib import pyplot as plt


def main():
    NOOP = np.asarray([[0, 0, 0, 0],
                       [0, 0, 0.0348, 0],
                       [0, 0, 0.034, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0.0466, 0],
                       [0, 0.0339, 0.318, 0.0168]])

    LEFT = np.asarray([[0, 0, 0, 0],
                       [0, 0, 0.0348, 0],
                       [0, 0, 0.034, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0.0267, 0],
                       [0, 0, 0.3557, 0]])

    RIGHT = np.asarray([[0, 0, 0, 0],
                        [0, 0, 0.0348, 0],
                        [0, 0, 0.034, 0],
                        [0, 0, 0.0971, 0],
                        [0, 0, 0.3195, 0],
                        [0, 0, 0.6276, 0]])

    GAS = np.asarray([[0, 0, 0, 0],
                      [0, 0, 0.0348, 0],
                      [0, 0, 0.034, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0.2708, 0],
                      [0.0746, 0, 0.4651, 0]])

    BRAKE = np.asarray([[0, 0, 0, 0],
                        [0, 0, 0.0348, 0],
                        [0, 0, 0.0617, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0.2355, 0]])

    predictions = (LEFT, RIGHT, GAS, BRAKE, NOOP)
    pred_names = ('Left', 'Right', 'Gas', 'Brake', 'NO-OP')

    plt.style.use('seaborn-colorblind')
    plot_all(predictions, pred_names)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()

    noop_gas = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.0574, 0],
        [0, 0, 0, 0],
        [0.0029, 0, 0.1105, 0],
        [0, 0.0045, 0.1308, 0.0412]])

    noop_left = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.0702, 0],
        [0, 0, 0, 0],
        [0.039, 0, 0.0947, 0],
        [0, 0, 0.1487, 0.043]])

    left_gas = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.0574, 0],
        [0, 0, 0, 0],
        [0, 0, 0.2073, 0],
        [0, 0, 0.4486, 0.0412]])

    left_left = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.0702, 0],
        [0, 0, 0, 0],
        [0, 0, 0.2573, 0],
        [0, 0, 0.5146, 0.043]])

    right_gas = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0.0801, 0],
        [0, 0, 0.0574, 0.1017],
        [0, 0, 0.1122, 0.0842],
        [0, 0, 0.2098, 0.094],
        [0, 0, 0.3873, 0.0412]])

    right_left = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0.0678, 0],
        [0, 0, 0.0702, 0.0517],
        [0, 0, 0.1473, 0.0646],
        [0, 0, 0.2015, 0.0506],
        [0, 0, 0.3433, 0.043]])

    gas_gas = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0.0235, 0],
        [0, 0, 0.0574, 0],
        [0, 0, 0, 0],
        [0, 0, 0.2273, 0.0096],
        [0, 0.0235, 0.4703, 0.0412]])

    gas_left = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0.0221, 0],
        [0, 0, 0.0702, 0],
        [0, 0, 0, 0],
        [0, 0, 0.2545, 0.1098],
        [0, 0.016, 0.5045, 0.043]])

    brake_gas = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.0574, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.1297, 0.0412]])

    brake_left = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.0702, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.1707, 0.043]])

    gas = (left_gas, right_gas, gas_gas, brake_gas, noop_gas)
    left = (left_left, right_left, gas_left, brake_left, noop_left)

    plot_double(gas, left, pred_names)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()

    LEFT_collision = np.asarray(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0.0829, 0],
         [0, 0, 0, 0],
         [0, 0, 0.1619, 0],
         [0, 0, 0.55, 0]])

    RIGHT_collision = np.asarray(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0.0829, 0],
         [0.0187, 0, 0, 0],
         [0, 0, 0.1619, 0],
         [0, 0, 0.5232, 0]])

    GAS_collision = np.asarray(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0.0829, 0],
         [0, 0, 0, 0],
         [0, 0, 0.2511, 0],
         [0, 0, 0.7075, 0]])

    BRAKE_collision = np.asarray(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0.0829, 0],
         [0, 0, 0, 0],
         [0, 0, 0.1619, 0],
         [0, 0, 0, 0]])

    NOOP_collision = np.asarray(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0.0829, 0],
         [0, 0, 0, 0],
         [0, 0, 0.1619, 0],
         [0, 0, 0.3001, 0]])

    predictions_collision = (LEFT_collision, RIGHT_collision, GAS_collision, BRAKE_collision, NOOP_collision)
    plot_all(predictions_collision, pred_names)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()


def plot_all(predictions, names):
    temp_offset = [8, 16, 32]
    goal = np.array([1, 0, 1, -1])
    temp_coef = np.array([0, 0, 0, 0.5, 0.5, 1])

    fig, axs = plt.subplots(3, 2, figsize=(8, 10), sharey=True)

    for idx, prediction, name in zip(np.ndindex(3, 2), predictions, names):
        for pred, to in zip((prediction * goal * temp_coef[:, np.newaxis])[::-1], temp_offset[::-1]):
            axs[idx].bar([1, 2, 3, 4], pred, label=to)
            axs[idx].plot([0, 5], [0, 0], 'k--', linewidth=0.8)

        val = np.sum(prediction * goal * temp_coef[:, np.newaxis])
        axs[idx].text(0.05, 0.95, round(val, 4), transform=axs[idx].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      weight=('bold' if idx == (1, 0) else 'normal'))

        axs[idx].set_title(f'Action: {name}')
        axs[idx].set_xticks([1, 2, 3, 4])
        axs[idx].set_xticklabels(['Distance', 'Speed', 'Tiles', 'Crashes'])

    axs[2, 1].set_visible(False)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], prop={'size': 10}, bbox_to_anchor=(0.53, 0.294), loc='upper left', title='Temporal offset')

    fig.suptitle('Future DFP predictions')


def plot_double(predictions_1, predictions_2, names):
    temp_offset = [8, 16, 32]
    goal = np.array([1, 0, 1, -1])
    temp_coef = np.array([0, 0, 0, 0.5, 0.5, 1])

    fig, axs = plt.subplots(5, 2, figsize=(8, 14), sharey=True)

    for y, x in np.ndindex(5, 2):
        prediction = (predictions_1 if x == 0 else predictions_2)[y]
        prediction = prediction * goal * temp_coef[:, np.newaxis]

        for pred, to in zip(prediction[::-1], temp_offset[::-1]):
            axs[y, x].bar([1, 2, 3, 4], pred, label=to)
            axs[y, x].plot([0, 5], [0, 0], 'k--', linewidth=0.8)

        val = np.sum(prediction)
        axs[y, x].text(0.05, 0.95, round(val, 4), transform=axs[y, x].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       weight=('bold' if (y, x) in [(0, 1), (2, 0)] else 'normal'))

        axs[y, x].set_xticks([1, 2, 3, 4])
        axs[y, x].set_xticklabels(['Distance', 'Speed', 'Tiles', 'Crashes'])
        axs[y, x].yaxis.set_label_position('right')
        if x == 1: axs[y, x].set_ylabel(names[y])

    axs[0, 0].set_title(f'\nPredicted action: Gas')
    axs[0, 1].set_title(f'\nPredicted action: Left')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles[::-1], labels[::-1], prop={'size': 10}, title='Temporal offset')
    fig.suptitle('Future DFP predictions')


if __name__ == '__main__':
    main()
