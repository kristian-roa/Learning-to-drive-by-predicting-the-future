import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu as MWU
import numpy as np

# MODELS:
# DFP:
# Mar-26 22.17 - Obstacles 2.5m
# Mar-26 02.05 - No obstacles 1m
#
# DQN:
# Feb-24 22.28 - Obstacles 2.5m
# Feb-26 11.40 - No obstacles 1m


def read(scenario, obstacles):
    # dfp
    dfp = pd.read_csv(f'DFP/Test/agent_4/scenario_{scenario}/Mar-26/dfp_stats_22.17' +
                      (f'_obs={obstacles}' if scenario in [0, 4] else '') + '.csv')

    # dfp no obst
    dfp_no = pd.read_csv(f'DFP/Test/agent_4/scenario_{scenario}/Mar-26/dfp_stats_02.05' +
                         (f'_obs={obstacles}' if scenario in [0, 4] else '') + '.csv')

    # dqn
    dqn = pd.read_csv(f'DQN/Test/agent_4/scenario_{scenario}/Feb-24/dqn_stats_22.28' +
                      (f'_obs={obstacles}' if scenario in [0, 4] else '') + '.csv')

    # dqn no obst
    dqn_no = pd.read_csv(f'DQN/Test/agent_4/scenario_{scenario}/Feb-26/dqn_stats_11.40' +
                         (f'_obs={obstacles}' if scenario in [0, 4] else '') + '.csv')

    return dfp, dfp_no, dqn, dqn_no


def read_3():
    # dfp scenario 3
    dfp_3 = pd.read_csv(f'DFP/Test/agent_3/scenario_3/Mar-27/dfp_stats_13.36.csv')

    # dqn scenario 3
    dqn_3 = pd.read_csv(f'DQN/Test/agent_3/scenario_3/Mar-01/dqn_stats_11.39.csv')

    return dfp_3, dqn_3


def make_plots(scenario, obstacles, metric):
    dfp, dfp_no, dqn, dqn_no = read(scenario, obstacles)

    data = pd.DataFrame()
    data['Episodes'] = range(1, 101)
    data['DFP'] = dfp[metric]
    data['DFP means'] = dfp[metric].rolling(20, min_periods=1).mean()
    data['DFP NO'] = dfp_no[metric]
    data['DFP NO means'] = dfp_no[metric].rolling(20, min_periods=1).mean()
    data['DQN'] = dqn[metric]
    data['DQN means'] = dqn[metric].rolling(20, min_periods=1).mean()
    data['DQN NO'] = dqn_no[metric]
    data['DQN NO means'] = dqn_no[metric].rolling(20, min_periods=1).mean()

    title = f'Scenario {scenario}' + (f', obstacles: {obstacles}' if scenario in [0, 4] else '')

    plt.figure()
    ax = data.boxplot(['DFP', 'DFP NO', 'DQN', 'DQN NO'], grid=False)
    ax.set_title(title)
    ax.set_ylabel(metric)
    plt.savefig(f'combined_plots/scenario_{scenario}' + (f'obs={obstacles}' if scenario in [0, 4] else '') + ('_reward' if metric == 'Rewards' else '') + '.png')


def make_plots_3(metric):
    dfp_3, dqn_3 = read_3()

    data_3 = pd.DataFrame()
    data_3['Episodes'] = range(1, 101)
    data_3['DFP'] = dfp_3[metric]
    data_3['DFP means'] = dfp_3[metric].rolling(20, min_periods=1).mean()
    data_3['DQN'] = dqn_3[metric]
    data_3['DQN means'] = dqn_3[metric].rolling(20, min_periods=1).mean()

    plt.figure()
    ax = data_3.boxplot(['DFP', 'DQN'], grid=False)
    ax.set_title('Scenario 3')
    ax.set_ylabel(metric)
    plt.savefig('combined_plots/scenario_3' + ('_reward' if metric == 'Rewards' else '') + '.png')


def update_all(metric):
    for scenario, obstacles in [(0, 0), (0, 3), (1, 0), (2, 0), (4, 0), (4, 3)]:
        make_plots(scenario, obstacles, metric)

    make_plots_3(metric)


def mannwhitneyu(scenario, obstacles, metric='Tiles'):
    dfp, dfp_no, dqn, dqn_no = read(scenario, obstacles)

    data = pd.DataFrame()
    data['Episodes'] = range(1, 101)
    data['DFP'] = dfp[metric]
    data['DFP NO'] = dfp_no[metric]
    data['DQN'] = dqn[metric]
    data['DQN NO'] = dqn_no[metric]

    keys = ('DFP', 'DFP NO', 'DQN', 'DQN NO')

    p_values = np.zeros(shape=(4, 4))
    for i, main in enumerate(keys, start=1):
        for j, key in enumerate(keys, start=1):
            statistic, p_value = MWU(data[main], data[key], alternative='two-sided')
            p_values[i-1, j-1] = p_value

    colors = np.where((p_values < 0.05) == 1, 0.3, 0.9)
    colors = np.where((p_values < 0.0125) == 1, 0.05, colors)
    p_value_text = np.empty(shape=(4, 4), dtype=object)
    for (y, x), val in np.ndenumerate(p_values):
        p_value_text[y, x] = f'{val:{".3e" if val < 1e-3 else ".4" if val < 1 else ""}}'

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=p_value_text,
                     rowLabels=keys,
                     colLabels=keys,
                     cellColours=plt.cm.RdYlBu(colors * 0.9),
                     loc='center',
                     colWidths=[0.1 for _ in keys])
    table.scale(2, 2)
    fig.tight_layout()
    plt.savefig(f'combined_plots/table_scenario_{scenario}' + (f'obs={obstacles}' if scenario in [0, 4] else '') + ('_reward' if metric == 'Rewards' else '') + '.png')


def main():
    scenario = 0
    obstacles = 3
    metric = 'Tiles'  # Tiles or Rewards
    # metric = 'Rewards'  # Tiles or Rewards

    # update_all(metric)
    mannwhitneyu(scenario, obstacles, metric)


if __name__ == '__main__':
    main()
