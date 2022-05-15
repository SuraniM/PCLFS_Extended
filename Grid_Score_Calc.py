import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import Constants as const


def get_best_grid_score(grid_scores, n_pclfs_features, step, model, plot_true, smote, source):

    n_features = len(grid_scores)

    # grid_threshold = 0.001*1.5
    max_f1 = max(grid_scores)
    print(grid_scores)
    print(n_pclfs_features)
    grid_scores_new = grid_scores[:n_pclfs_features-1]
    grid_threshold = 0.05/n_features  # 0.05 is the maximum F1-score deduction

    search_peak = find_peaks(grid_scores_new, height=0)
    peaks = search_peak[0]
    peak_heights = list(list(search_peak[1].values())[0])
    peaks = list(peaks+1)

    if grid_scores[0] > grid_scores[1]:
        peaks.insert(0, 1)
        peak_heights.insert(0, grid_scores[0])

    grid_score_df = pd.DataFrame({'features': peaks, 'f1_scores': peak_heights})

    decision, decision2 = [], []

    for i in range(len(peak_heights)):
        a = max_f1 - grid_score_df['f1_scores'][i]
        b = step*(n_pclfs_features - grid_score_df['features'][i])
        rule = (a/b)
        decision.append(rule)

        if decision[i] <= grid_threshold:
            decision2.append(True)
        else:
            decision2.append(False)

    grid_score_df['decision'] = decision
    grid_score_df['decision2'] = decision2

    select = grid_score_df[grid_score_df["decision2"]]

    if not select.empty:
        select.sort_values(by=['features'])
        select_features = select['features'].iloc[0]
        select_f1_score = select['f1_scores'].iloc[0]
    else:
        select_features = n_pclfs_features
        select_f1_score = max_f1
    old_f1_score = max_f1
    local_max = 0

    if n_pclfs_features == select_features:
        changed_result = False
    else:
        changed_result = True

    output = {'n_select_features': select_features, 'select_f1_score': select_f1_score, 'old_f1_score': old_f1_score,
          'grid_threshold': np.round(grid_threshold, 4), 'local_max': local_max, 'changed_result': changed_result}

    if plot_true:
        plt.rcParams.update({'font.size': 15})
        x = range(1, n_features+1, 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        if smote:
            title = model + " with SMOTE - t = " + str(np.round(grid_threshold, 4))
        else:
            title = model + " without SMOTE - t = " + str(np.round(grid_threshold, 4))
        plt.title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = fig.add_subplot(111)  # PCA
        plt.plot(x, grid_scores, color='blue')
        plt.plot(x, grid_scores, 'red', markevery=[x.index(n_pclfs_features)], ls="",
                 marker="o",
                 label="PCLFS " + '({}, {})'.format(n_pclfs_features, round(max_f1, 4)))
        plt.plot(x, grid_scores, 'blue', markevery=[x.index(select_features)], ls="", marker="o",
                 label="Ext_PCLFS " + '({}, {})'.format(select_features, round(select_f1_score, 4)))
        plt.plot(n_pclfs_features, max_f1, 'red', ls="", marker="o")
        # plt.axvline(x=10, color='k', linestyle='--')
        plt.ylim((0.0, 1))
        plt.ylabel('F1-score')
        plt.xlabel('Number of selected features')
        plt.legend()
        plt.savefig(const.Constants.SAVE_PATH + source + '\\images\\' + str(title) + '.png', dpi=300)
        plt.show()

    return output
