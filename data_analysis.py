
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_sec(time_str):
    h, m, s = time_str.split(':')
    return float(int(h) * 3600 + int(m) * 60 + int(s))


def data_avrg(df):
    """
    """
    # t_range = np.arange(330, len(df), 60)

    # hours = [[] for _ in range(16)]
    # i = 0
    # for strt in t_range:
    #     end = strt + 60
    #     # print(i, strt, end)
    #     if i < 16:
    #         hours[i].append(df.iloc[strt:end, :]['valor'].mean())
    #     i += 1
    #     if i == 24:
    #         i = 0

    # labels = ["6:00", "7:00", "8:00", "9:00", "10:00", "11:00", "12:00", "13:00",
    #           "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00",
    #           "21:00"]

    m_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # t_range = np.arange(660, len(df), 180)
    months, strt = [], 0
    for m in m_days:
        # print(m)
        end = strt + 60 * 24 * m
        months.append(df.iloc[strt:end, :]['valor'])
        strt = end

    return months


# Create a dataframe
df = pd.read_csv("input/mer_uvb_ipina.csv").replace([-9.999], [0.000])
df['valor'] = df['valor'] * .0583
print(df.head())

months = data_avrg(df)

with sns.axes_style("darkgrid"):

    # figsize(x1, y1), GridSpec(y2, x2)
    fig = plt.figure(figsize=(30, 60))
    gs = gridspec.GridSpec(12, 2)

    for i in range(11):
        ax = plt.subplot(gs[i:(i + 1), 0:2])
        # plt.ylim(0., .5)
        plt.ylabel("UV")
        plt.plot(months[i], lw=.6)
        plt.legend(loc='upper left')

# plt.show()
fig = ax.get_figure()
fig.savefig("output/test.png", dpi=150, bbox_inches='tight')
import pdb; pdb.set_trace()  # breakpoint c9ffae3c //


ax = sns.kdeplot(df.valor[df.valor > 0])
ax.set(xlabel='valor')
# sns.plt.show()

fig = ax.get_figure()
fig.savefig("output/test.png", dpi=150)

import pdb; pdb.set_trace()  # breakpoint 8aea782e //

t_sec = get_sec(df.time)
import pdb; pdb.set_trace()  # breakpoint d95faa0e //


# ggplot(df, aes(x="valor", weight="time")) + geom_bar(fill='blue')
