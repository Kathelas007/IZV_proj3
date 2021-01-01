import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np
import seaborn as sns
import datetime
import astral
import astral.geocoder
import astral.sun
from scipy._lib.decorator import getfullargspec


def convert_date(df: pd.DataFrame):
    df['p2a'] = pd.to_datetime(df['p2a'])
    return df.rename(columns={'p2a': 'date'})


def avg_accidents_per_day(df: pd.DataFrame):
    count_per_days_df = df[['p1', 'date']].groupby('date').count()
    avg_per_day = count_per_days_df['p1'].mean()
    return avg_per_day


def fig_accidents_during_week(df, p5a=1):
    day_names = ["PO", "ÚT", "ST", "ČT", "PÁ", "SO", "NE"]
    titles = ['V obci', 'Mimo obec']

    axes = []
    figs = []
    max_y = 0

    sns.set_style("darkgrid")

    # create plots
    for i in range(2):
        df1 = df[df['p5a'] == i + 1]

        number_of_dates_in_df = len(df1['date'].unique())
        accidents_per_week_day_df = df1[['p1', 'weekday(p2a)']].groupby('weekday(p2a)').count().reset_index()
        accidents_per_week_day_df['p1'] = accidents_per_week_day_df['p1'] * 365 / number_of_dates_in_df

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        sns.barplot(data=accidents_per_week_day_df, y='p1', x=day_names)

        axes.append(ax)
        figs.append(fig)
        max_y = max(max_y, accidents_per_week_day_df['p1'].max())

    # style plots
    for i in range(2):
        axes[i].set_ylabel('počet')
        axes[i].set_title(titles[i])
        # axes[i].set_ylim(top=max_y * 1.2)

        figs[i].savefig(f'weekday_{i + 1}.png')

    plt.show()
    plt.close()


def fig_holidays(df):
    # full years only
    df = df[df['date'] < pd.to_datetime(datetime.date(year=2020, month=1, day=1))]

    # create dates
    new_years = [datetime.date(year=i, month=1, day=1) for i in range(2016, 2020)]
    easter_mondays = [datetime.date(year=2016, month=3, day=28), datetime.date(year=2017, month=4, day=17),
                      datetime.date(year=2018, month=4, day=2), datetime.date(year=2019, month=4, day=22)]
    christmas = [datetime.date(year=i, month=12, day=24) for i in range(2016, 2020)]

    # get avg counts for holidays
    holidays = [new_years, easter_mondays, christmas]
    counts = []
    for holiday in holidays:
        count = df[df['date'].isin(holiday)]['p1'].count() / 4
        counts.append(count)

    sns.set_style("darkgrid")
    x_names = ["Nový rok", "Velikonoční\npondělí", "Vánoce"]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # plot bars
    sns.barplot(ax=ax, x=x_names, y=counts)

    # plot avg
    avg = [avg_accidents_per_day(df)] * 3
    sns.lineplot(ax=ax, x=x_names, y=avg)

    fig.savefig(f'holidays.png')

    plt.show()
    plt.close()


#
# def add_time_to_date(x):
#     return x['date'].replace(hour=x['hours'], minute=x['minutes'])


city = astral.geocoder.lookup("Prague", astral.geocoder.database())


def get_sunrise_sunset(date):
    astral_data = astral.sun.sun(city.observer, date=date, tzinfo='Europe/Prague')
    return [astral_data['sunrise'], astral_data['sunset']]


def tab_sunrise_sunset(df):
    # full years only
    df = df[df['date'] < pd.to_datetime(datetime.date(year=2020, month=1, day=1))]

    # prepare
    df1 = df[['date', 'p2b', 'p1']].groupby(['date', 'p2b']).count().reset_index()
    df1.rename(columns={'p1': 'count'}, inplace=True)
    df1['date'] = df1['date'].dt.tz_localize(tz='Europe/Prague')

    # get sunset / sunrise
    df_ss = pd.DataFrame(df1['date'].unique(), columns=['date'])
    df_ss['sunrise_sunset'] = df_ss['date'].apply(get_sunrise_sunset)
    df_ss['sunrise'] = df_ss['sunrise_sunset'].apply(lambda x: x[0])
    df_ss['sunset'] = df_ss['sunrise_sunset'].apply(lambda x: x[1])
    df_ss.drop(['sunrise_sunset', ], axis=1, inplace=True)

    # merge by date
    dfm = df1.merge(df_ss, left_on='date', right_on='date')

    # get full hours, minutes
    dfm['hours'] = dfm['p2b'].str.slice(0, 2).astype(int)
    dfm['minutes'] = dfm['p2b'].str.slice(2, None).astype(int)
    dfm.drop(['p2b', ], axis=1, inplace=True)
    # remove bad non-sence time
    dfm = dfm[(dfm['hours'] < 24) & (dfm['minutes'] < 60)]

    # add time to date
    dfm['date'] = dfm['date'] + pd.to_timedelta(dfm['minutes'], 'm') + pd.to_timedelta(dfm['hours'], 'h')
    dfm.drop(['hours', 'minutes'], axis=1, inplace=True)

    # mark sunset(1) /sunrise(2) / other(0)
    dfm['sunrise_diff'] = dfm['sunrise'] - dfm['date']
    dfm['sunset_diff'] = dfm['date'] - dfm['sunset']

    td_0 = pd.to_timedelta(0, 'h')
    td_2 = pd.to_timedelta(2, 'h')
    dfm['daytime'] = ((dfm['sunrise_diff'] > td_0) & (dfm['sunrise_diff'] < td_2)) * 1 + \
                     ((dfm['sunset_diff'] > td_0) & (dfm['sunset_diff'] < td_2)) * 2
    # dfm[['date', 'sunrise', 'sunrise_diff']]

    print('a')


def generate_data(df: pd.DataFrame):
    print(f'Average accident count per day:\t{avg_accidents_per_day(df)}')
    # fig_accidents_during_week(df)
    # fig_holidays(df)
    tab_sunrise_sunset(df)


if __name__ == "__main__":
    df = pd.read_pickle("accidents.pkl.gz")
    df = convert_date(df)
    generate_data(df)
