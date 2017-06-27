
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def loadData():
    """
    """
    # df = pd.read_csv("input/2002clearsky/2002UVBclearsky.txt", sep='\t')
    df = pd.read_csv("input/2002clearsky/2002clearskyUVA.txt", sep='\t')
    # Convert '--' to NaN
    df = df.convert_objects(convert_numeric=True)
    for st in ['FAC', 'HAN', 'MER', 'MON', 'PED', 'SAG', 'TLA']:
        df.loc[df[st] < 0., st] = np.nan
    print(df.head())

    # Group by days.
    # data = df.groupby(['Fecha'])
    data = df.groupby(['Date'])

    return data


def main():
    """
    """

    data = loadData()
    station_list = ['FAC', 'HAN', 'MER', 'MON', 'PED', 'SAG', 'TLA']

    fig = plt.figure(figsize=(20, 100))
    gs = gridspec.GridSpec(20, 4)

    c = 0
    for name, group in data:
        print(name)
        hours = group.query('5 < Hora < 21')['Hora']
        h_range = group.query('5 < Hora < 21').ix[:, 'FAC':'TUVmodel']
        # Indexes of hours that define the range.
        idx1 = hours[hours == 12.5].index[0]
        idx2 = hours[hours == 13.5].index[0]

        ax = fig.add_subplot(gs[c])
        area_s, y_max = [], []
        plt.title(name)
        plt.xlabel("Hours")
        for station in station_list:
            # print(h_range[station])
            plt.plot(hours, h_range[station], label=station, lw=1.5, ls='--')
            y1s, y2s = h_range[station][idx1], h_range[station][idx2]
            area_s.append((y2s + y1s) / 2.)
            y_max.append(max(h_range[station]))

        plt.plot(hours, h_range['TUVmodel'], lw=1.5, label='TUV')
        y_max.append(np.nanmax(np.array(h_range['TUVmodel'])))
        plt.ylim(0., np.nanmax(y_max) + np.nanmax(y_max) * .1)
        plt.legend()

        y1, y2 = h_range['TUVmodel'][idx1], h_range['TUVmodel'][idx2]
        area = (y2 + y1) / 2.
        areas_coeff = (area - area_s) / area

        ax = fig.add_subplot(gs[c + 1])
        mednan, meanan = np.nanmedian(areas_coeff), np.nanmean(areas_coeff)
        ax.axhline(
            mednan, ls='--', color='g',
            label=r"$Median\approx{:.2f}$".format(mednan))
        ax.axhline(
            meanan, ls='--', color='r',
            label=r"$Mean\approx{:.2f}$".format(meanan))
        plt.scatter(range(len(station_list)), areas_coeff)
        ax.set_xticks(range(len(station_list)))
        ax.set_xticklabels(station_list)
        plt.legend()

        c += 2

    print("\nPlotting")
    fig.tight_layout()
    plt.savefig('output/clearsky_UVA.png', dpi=150, bbox_inches='tight')
    print("Finished.")


if __name__ == '__main__':
    main()
