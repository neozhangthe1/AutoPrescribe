import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')
# mpl.rc('text', color='black')

import numpy as np
from collections import defaultdict as dd
from utils.data import dump, load
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def get_data():
    dataset = "sutter"
    level = 2
    import os
    results = dd(list)
    xs = {}
    for file in os.listdir("/home/yzhang3151/project/AutoPrescribe2/data"):
        if file.endswith(".pkl") and file.startswith("%s_%s" % (dataset, level)):
            d = file.split("_")
            results[d[2]].append((int(d[5][1:]), int(d[6][1:]), float(d[7].replace("jacc", ""))))

    for k in results:
        xs[k] = [y[2] for y in sorted(results[k], key=lambda x: (x[0], x[1]))]

    dump(xs, "traj_%s_%s.pkl" % (dataset, level))

    xs = load("traj_%s_%s.pkl" % (dataset, level))

    name_mapping = {
        "voc": "Vocabulary",
        "random": "Random",
        "freq": "Frequent first",
        "rare": "Rare first"
    }


    fig, ax = plt.subplots(figsize=(8, 4))

    for k in xs:
        line, = ax.plot([x*1.35 for x in xs[k]][:len(xs["random"])], '-', linewidth=2, label=name_mapping[k])

    # x = np.linspace(0, 10, 500)
    # dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
    #
    # line1, = ax.plot(x, np.sin(x), '--', linewidth=2,
    #                  label='Dashes set retroactively')
    # line1.set_dashes(dashes)
    #
    # line2, = ax.plot(x, -1 * np.sin(x), dashes=[30, 5, 10, 5],
    #                  label='Dashes set proactively')
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Jaccard Coefficient", fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax.legend(loc='lower right', fontsize=20)
    # plt.show()
    fig.tight_layout()
    plt.savefig("traj_%s_%s.pdf" % (dataset, level))
