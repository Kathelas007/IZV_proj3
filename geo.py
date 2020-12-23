#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np


# todo crs vyresit


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    # EPSG: 5514

    df = df[(~df["d"].isin([np.nan, np.inf, -np.inf])) & (~df["d"].isin([np.nan, np.inf, -np.inf]))]
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df["d"], df["e"]), crs="EPSG:5514")
    gdf = gdf.drop(["d", "e"], axis=1)
    return gdf


selected_reg = "ZLK"


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s dvemi podgrafy podle lokality nehody """
    gdf = gdf[gdf["region"] == selected_reg]

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
    # todo crs ??
    gdf = gdf[gdf["region"] == selected_reg]
    # gdf = gdf.to_crs(epsg=3857)

    accidents = gdf.geometry

    # get model
    coords = np.dstack([gdf.geometry.x, gdf.geometry.y]).reshape(-1, 2)
    model = sklearn.cluster.MiniBatchKMeans(n_clusters=40)
    db = model.fit(coords)

    # mng data
    gdf["cls_lab"] = db.labels_
    gdf = gdf.dissolve(by="cls_lab", aggfunc={"p1": "count"})

    centers = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(db.cluster_centers_[:, 0], db.cluster_centers_[:, 1]),
        crs=gdf.crs.to_string())

    gdf_plot = gdf.merge(centers, left_on="cls_lab", right_index=True).rename(columns={"p1": "count"}). \
        set_geometry("geometry_y")

    # gdf_plot = gdf_plot.to_crs("epsg:3857")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    accidents.plot(ax=ax, markersize=3, color="tab:purple", alpha=0.3)
    gdf_plot.plot(ax=ax, markersize=gdf_plot["count"], column="count", legend=True, alpha=0.6)
    ctx.add_basemap(ax, crs="epsg:5514", source=ctx.providers.Stamen.TonerLite, zoom=10, alpha=1)

    ax.set(ylabel='', xlabel='')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title("Nehody ve Zlínském kraji")

    for pos in ["top", "bottom", "right", "left"]:
        ax.spines[pos].set_visible(False)

    if show_figure:
        plt.show()
        plt.close()

    if fig_location is not None:
        fig.savefig(fig_location)


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    # plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
