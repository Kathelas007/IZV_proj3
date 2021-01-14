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
    """
    Fnc convert column in dataframe PD date
    :param df: dataFrame
    :return: changed dataFrame
    """
    df['p2a'] = pd.to_datetime(df['p2a'])
    return df.rename(columns={'p2a': 'date'})


def avg_accidents_per_day(df: pd.DataFrame):
    """
    Fnc counts average accidents count per one day
    :param df: inout dataFrame
    :return: result
    """
    count_per_days_df = df[['p1', 'date']].groupby('date').count()
    avg_per_day = count_per_days_df['p1'].mean()
    return avg_per_day


yellow = '#fdb705'


# yellow = '#fdbb05'
# yellow = '#fdbf05'


# palette = sns.color_palette("bright")


def make_tick_labels_white(xtick_labels, ytick_labels):
    for tick in xtick_labels + ytick_labels:
        tick.set_color('#ffffff')


def fig_accidents_during_week(df: pd.DataFrame):
    """
    Function creates figures showing accident count during week
    :param df: input dataFrame
    """
    day_names = ['PO', 'ÚT', 'ST', 'ČT', 'PÁ', 'SO', 'NE']
    df1 = df[['p5a', 'p1', 'weekday(p2a)']].groupby(['p5a', 'weekday(p2a)']).count().reset_index()
    df1['p1'] = df1['p1'] / (365)

    # replace
    str_range = [str(i) for i in range(7)]
    days = dict(zip(str_range, day_names))
    locations = {1: 'V obci', 2: 'Mimo obec'}
    df1 = df1.replace({'p5a': locations, 'weekday(p2a)': days})
    df1.rename(columns={'p1': 'počet', 'p5a': 'lokalita'}, inplace=True)

    # set style
    sns.set_style('darkgrid', {'text.color': 'white'})
    palette = sns.set_palette(sns.color_palette(['tab:red', 'tab:blue']))

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3))
    sns.barplot(data=df1, y='počet', hue='lokalita', x='weekday(p2a)', palette=palette)

    # axes style
    ax.set_ylabel('počet', color='#ffffff')
    ax.set_xlabel(None)
    ax.set_ylim(top=200)
    make_tick_labels_white(ax.get_xticklabels(), ax.get_yticklabels())
    ax.get_legend().set_title(None)
    fig.savefig(f'week.png', transparent=True)

    plt.show()
    plt.close()


def count_alcohol_percent_on_new_year(df: pd.DataFrame):
    ny_df = df[df['holidays'] == 'Nový rok']

    total = len(ny_df.index)
    # alcohol_count = len(ny_df[ny_df['p57'].isin([4, 5])].index)
    alcohol_count = len(ny_df[ny_df['p11'].isin([1, 3, 5, 6, 7, 8, 9])].index)

    return alcohol_count / (total / 100)


def fig_holidays(df: pd.DataFrame):
    """
    Fnc creates figure showing average accident count at New Year, Christmas and Easter Monday
    :param df: input dataFrame
    """

    # full years only
    df = df[df['date'] < pd.to_datetime(datetime.date(year=2020, month=1, day=1))]

    # create dates
    new_years = [datetime.date(year=i, month=1, day=1) for i in range(2016, 2020)]
    easter_mondays = [datetime.date(year=2016, month=3, day=28), datetime.date(year=2017, month=4, day=17),
                      datetime.date(year=2018, month=4, day=2), datetime.date(year=2019, month=4, day=22)]
    christmas = [datetime.date(year=i, month=12, day=24) for i in range(2016, 2020)]

    # get avg counts for holidays
    holidays = [new_years, easter_mondays, christmas]
    labels = ['Nový rok', 'Velikonoční pondělí', 'Vánoce']

    # mark holidays
    df['holidays'] = np.nan
    for holiday, label in zip(holidays, labels):
        df.loc[df['date'].isin(holiday), "holidays"] = label

    # mark alcohol
    alcohol_ny = count_alcohol_percent_on_new_year(df)

    # count number of occurenc
    df_plot = df[['holidays', 'p1']].groupby(['holidays']).count()
    df_plot['p1'] = df_plot['p1'] / 4
    df_plot.rename(columns={'p1': 'počet'}, inplace=True)

    # set styles
    sns.set_style('darkgrid', {'text.color': 'white'})
    palette = sns.set_palette(sns.color_palette(['tab:blue', 'tab:green', 'tab:red']))

    x_names = ['Nový rok', 'Velikonoční\npondělí', 'Vánoce']
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.6))

    # plot bars
    sns.barplot(ax=ax, x=x_names, y=df_plot['počet'], palette=palette)

    # style axis
    ax.set_ylabel('počet', color='#ffffff')
    make_tick_labels_white(ax.get_xticklabels(), ax.get_yticklabels())

    fig.savefig(f'holidays.png', transparent=True)

    plt.show()
    plt.close()

    return alcohol_ny


city = astral.geocoder.lookup('Prague', astral.geocoder.database())


def get_sunrise_sunset(date: datetime) -> list:
    """
    Fnc returns datetime of sunrise and sunset at given date
    :param date: date for sunrise, sunset
    :return: [sunrise, sunset]
    """
    astral_data = astral.sun.sun(city.observer, date=date, tzinfo='Europe/Prague')
    return [astral_data['sunrise'], astral_data['sunset']]


def tab_sunrise_sunset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fnc creates table of accident percent caused by wild animals during sunrise / sunset
    :param df: input data Frame
    :return: result dataframe
    """
    # full years only (2016-2019)
    df = df[df['date'] < pd.to_datetime(datetime.date(year=2020, month=1, day=1))]
    df = df[df['p5a'] == 2]  # lokalita mimo obec

    df['zver'] = df['p10'] == 4  # zavineni nehody lesni zveri

    # prepare, increase count by group by to speed up next calculations
    df1 = df[['date', 'p2b', 'p1', 'zver']].groupby(['date', 'p2b', 'zver']).count().reset_index()
    df1.rename(columns={'p1': 'count'}, inplace=True)
    df1['date'] = df1['date'].dt.tz_localize(tz='Europe/Prague')

    # get sunset / sunrise time
    df_ss = pd.DataFrame(df1['date'].unique(), columns=['date'])
    df_ss['sunrise_sunset'] = df_ss['date'].apply(get_sunrise_sunset)
    df_ss['sunrise'] = df_ss['sunrise_sunset'].apply(lambda x: x[0])
    df_ss['sunset'] = df_ss['sunrise_sunset'].apply(lambda x: x[1])
    df_ss.drop(['sunrise_sunset', ], axis=1, inplace=True)

    # merge sunset / sunrise with other data by date
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

    # add astronomical season
    dfm['season'] = ((dfm['date'] + pd.to_timedelta(20, 'D')).dt.month % 12 + 3) // 3

    # sount sunset / sunrise diff
    dfm['sunrise_diff'] = dfm['sunrise'] - dfm['date']
    dfm['sunset_diff'] = dfm['date'] - dfm['sunset']

    # mark sunset(1) /sunrise(2) / other(0)
    td_0 = pd.to_timedelta(0, 'h')
    td_2 = pd.to_timedelta(1, 'h')
    dfm['daytime'] = ((dfm['sunrise_diff'] > td_0) & (dfm['sunrise_diff'] < td_2)) * 1 + \
                     ((dfm['sunset_diff'] > td_0) & (dfm['sunset_diff'] < td_2)) * 2
    dfm = dfm[dfm['daytime'] > 0]

    df_res = dfm[['count', 'daytime', 'season', 'zver']].groupby(['daytime', 'season', 'zver']).sum().reset_index()

    # shape table properly
    num_to_seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    num_to_daytime = {1: 'Sunrise', 2: 'Sunset'}
    df_res = df_res.replace({'season': num_to_seasons, 'daytime': num_to_daytime})
    df_res = df_res.pivot(columns='zver', index=['season', 'daytime'], values='count')

    # counting percent
    df_res['Total'] = df_res[True] + df_res[False]
    df_res['Podil-zver'] = df_res[True] / (df_res['Total'] / 100)

    # create google shaped table
    df_tab = df_res['Podil-zver'].reset_index().pivot(columns='season', index='daytime', values='Podil-zver')
    return df_tab


def generate_data(df: pd.DataFrame):
    print(f'Average accident count per day:\t{avg_accidents_per_day(df)}')
    fig_accidents_during_week(df)
    print(f'Percent of alcohol on NY:\t{fig_holidays(df)}')
    print('\n', tab_sunrise_sunset(df))


if __name__ == '__main__':
    df = pd.read_pickle('accidents.pkl.gz')
    df = convert_date(df)
    generate_data(df)
