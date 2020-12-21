#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np


# muzeze pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    # EPSG: 5514
    df = df[(df["e"] != np.nan) & (df["d"] != np.nan)]
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df["d"], df["e"]), crs="EPSG:5514")
    gdf.drop(["d", "e"], axis=1)
    return gdf


selected_reg = "ZLK"


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s dvemi podgrafy podle lokality nehody """
    gdf = gdf[gdf["region"] == selected_reg]

    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Nehody ve Zlínském kraji", fontsize=16)
    fig.tight_layout()

    axes = axes.flatten()
    subtitles = ["V obci", "Mimo obec"]
    colors = ["lightseagreen", "orangered"]
    for i in range(2):
        gdf[gdf["p5a"] == i + 1].plot(ax=axes[i], markersize=3, color=colors[i])
        ctx.add_basemap(axes[i], crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite, zoom=10, alpha=1.6)

        axes[i].set(ylabel='', xlabel='')
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)
        axes[i].set_title(subtitles[i])

        for pos in ["top", "bottom", "right", "left"]:
            axes[i].spines[pos].set_visible(False)

    if show_figure:
        plt.show()
        plt.close()

    if fig_location is not None:
        fig.savefig(fig_location)


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
