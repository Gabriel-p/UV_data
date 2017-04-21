
import pandas as pd
import seaborn as sns


def get_sec(time_str):
    h, m, s = time_str.split(':')
    return float(int(h) * 3600 + int(m) * 60 + int(s))


# Create a dataframe
df = pd.read_csv("mer_uvb_ipina.csv")
print(df.head())

sns.kdeplot(df.valor[df.valor > 0])
sns.plt.show()


t_sec = get_sec(df.time)
import pdb; pdb.set_trace()  # breakpoint d95faa0e //


# ggplot(df, aes(x="valor", weight="time")) + geom_bar(fill='blue')
