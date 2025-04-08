# -*- coding: utf-8 -*-

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MANIFEST"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

"""

author: pietro devò
e-mail: pietro.devo@dicea.unipd.it

            .==,_
           .===,_`^
         .====,_ ` ^      .====,__
   ---     .==-,`~. ^           `:`.__,
    ---      `~~=-.  ^           /^^^
      ---       `~~=. ^         /
                   `~. ^       /
                     ~. ^____./
                       `.=====)
                    ___.--~~~--.__
          ___|.--~~~              ~~~---.._|/
          ~~~"                             /

"""

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""LIBRARIES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import functools
import itertools
import math
import numpy
import pandas

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import cross
import dates
import dists
import extraction
import file
import frames
import mev
import region
import stats
import utils
import validation

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def converter(
    filename,
    skip=True,
    remove=True,
    subdir=None,
    subext=None,
    subzip=None,
    subdic={},
    procdir=None,
    procext=None,
    proczip=None,
    procdic={},
    level=0,
):
    """converter function"""

    if subdir is None and procdir is not None:
        subdir = procdir
    if procdir is None and subdir is not None:
        procdir = subdir

    if file.flag(
        filename,
        subdir,
        subext,
    ):

        if skip and file.flag(
            filename,
            procdir,
            procext,
        ):

            utils.printing(
                "converter:",
                filename,
                "processing skipped",
                level=level,
            )

        else:
            try:
                utils.save(
                    utils.load(
                        filename, subdir, subext, subzip, level=1, **subdic
                    ),
                    filename,
                    procdir,
                    procext,
                    proczip,
                    level=1,
                    **procdic,
                )
                if remove:
                    file.remove(filename, subdir, subext)

            except:
                utils.printing(
                    "converter:",
                    filename,
                    "processing failed",
                    level=level,
                )
            else:
                utils.printing(
                    "converter:",
                    filename,
                    "processing success",
                    level=level,
                )


def datafetcher(
    station,
    pack=None,
    key=None,
    drop=None,
    function=None,
    arguments=(),
    keywords={},
    subdir="data",
    subext="csv",
    subzip=None,
    subdic={},
    level=0,
):
    """datafetcher function"""

    try:
        # loading
        entry = utils.load(station, subdir, subext, subzip, level=1, **subdic)

        if key is not None:
            if pack is None:
                entry = entry.loc[key]
            else:
                entry = entry[key]

        if drop is not None:
            entry.dropna(how=drop, inplace=True)

        if function is not None:
            if isinstance(function, str):
                function = utils.definition(function)
            elif isinstance(function, (list, tuple)):
                function = utils.definition(*function)
            elif isinstance(function, dict):
                function = utils.definition(**function)

            if callable(function):
                entry = function(entry, *arguments, **keywords)

        # indexing
        entry.loc[:, "code"] = station

    except:
        utils.printing(
            "datafetcher:", station, "retrieving failed", level=level
        )
        return
    else:
        utils.printing(
            "datafetcher:", station, "retrieving success", level=level
        )
        return entry


def stormwalker(
    station,
    frequency=None,
    filename=None,
    split=None,
    reader=None,
    skipspaces=True,
    skipheader=0,
    skipfooter=0,
    variables=None,
    dtype=None,
    separator=None,
    dateparser=True,
    dateformat="%Y-%m-%d %H:%M:%S",
    thousands=None,
    decimal=".",
    flags=None,
    nans=None,
    rename=None,
    index="datetime",
    keep=None,
    scale=None,
    continuous=None,
    resample=None,
    interpolate=None,
    sort=True,
    drop=None,
    readerdic={},
    column="value",
    blocks="y",
    percentile=0.5,
    length=None,
    events=None,
    eventsdata=None,
    eventskeys=None,
    pack=None,
    skip=None,
    out=None,
    subdir="series",
    subext="csv",
    subzip="infer",
    subdic={},
    procdir="extraction",
    proczip=None,
    procdic={},
    level=0,
    **keywords
):
    """storm events analyses"""

    if pack is None:
        inpext = "csv"
    else:
        inpext = "pkl"

    if eventsdata is not None and eventskeys is not None:
        if not isinstance(eventskeys, (list, tuple)):
            eventskeys = [eventskeys]

        if isinstance(eventsdata, dict):
            keywords.update(**{i: eventsdata[station][i] for i in eventskeys})
        elif type(eventsdata).__name__ == "DataFrame":
            keywords.update(
                **{i: eventsdata.loc[station, i] for i in eventskeys}
            )

    if skip and file.flag(station, procdir, inpext):
        utils.printing(
            "stormwalker:", station, "processing skipped", level=level
        )
        if out:
            return utils.load(
                station,
                procdir,
                inpext,
                proczip,
                level=1,
                **procdic,
            )

    else:
        try:
            if pandas.isna(filename):
                filename = station

            # series loading
            series = extraction.series(
                filename,
                subdir,
                subext,
                subzip,
                subdic,
                split,
                reader,
                skipspaces,
                skipheader,
                skipfooter,
                variables,
                separator,
                dateparser,
                dateformat,
                thousands,
                decimal,
                flags,
                rename,
                index,
                nans,
                dtype,
                keep,
                scale,
                continuous,
                resample,
                interpolate,
                sort,
                drop,
                **readerdic,
            )

            if pandas.isna(frequency):
                frequency = dates.frequency(series)
            elif resample is not None:
                frequency = resample

            # frequency formatter
            frequency = dates.formatter(
                frequency, spaces=0, digits=2, units=True, textcase="lower"
            )

            try:
                series = extraction.filtering(
                    series,
                    frequency,
                    column,
                    blocks,
                    percentile,
                    length,
                )
            except:
                utils.printing(
                    "stormwalker:", station, "insufficient data", level=level
                )
                raise

            if pack is None:
                outkey = "maxima"
                outext = "csv"
            else:
                outkey = True
                outext = "pkl"

            if events is None:
                output = series.replace(0, numpy.nan).rename(
                    columns={column: frequency}
                )
            else:
                output = utils.definition(extraction, events)(
                    series,
                    frequency,
                    column,
                    output=outkey,
                    **keywords,
                )

            if procdir:
                utils.save(
                    output,
                    station,
                    procdir,
                    outext,
                    proczip,
                    level=1,
                    **procdic,
                )

        except:
            utils.printing(
                "stormwalker:", station, "processing failed", level=level
            )
            if out:
                return
        else:
            utils.printing(
                "stormwalker:", station, "processing success", level=level
            )
            if out:
                return output


def clusters(
    dtb,
    values,
    station,
    target=None,
    projections=None,
    x="longitude",
    y="latitude",
    years=None,
    cumulative=None,
    dimension=None,
    radius=None,
    maxima=None,
    order="L2",
    partial=None,
    distribution="gev",
    arguments=(),
    keywords={},
    realizations=1e3,
    heterogeneity=None,
    ranking=None,
    infer=None,
    durations=None,
    indexcode="code",
    indexdatetime="datetime",
    lengths="length",
    pack=None,
    skip=None,
    out=None,
    subdir=None,
    subext="csv",
    subzip=None,
    subdic={},
    procdir="clusters",
    proczip=None,
    procdic={},
    level=0,
):
    """clusters definition function"""

    if pack is None:
        procext = "csv"
    else:
        procext = "pkl"

    if skip and file.flag(station, procdir, procext):
        if out:
            utils.printing(
                "clusters:", station, "processing resumed", level=level
            )
            # loading
            output = utils.load(
                station,
                procdir,
                procext,
                proczip,
                **procdic,
                level=1,
            )

            if pack is None:
                return output.set_index(
                    pandas.Index([station] * output.shape[0], name="code")
                )
            else:
                return output["cluster"]

        else:
            utils.printing(
                "clusters:", station, "processing skipped", level=level
            )
            return

    else:

        if isinstance(values, (str, list, tuple)):
            if subdir is None:
                subdir = values
            # defining datafetcher
            values = functools.partial(
                lambda filename, subdir, subext, subzip, subdic: (
                    lambda i: (
                        i
                        if i.index.names[0] == indexcode
                        else i.reset_index().set_index("code")
                    )
                )(
                    datafetcher(
                        filename,
                        subdir=subdir,
                        subext=subext,
                        subzip=subzip,
                        **subdic,
                        level=1,
                    )
                ),
                subdir=subdir,
                subext=subext,
                subzip=subzip,
                subdic=subdic,
            )

        if type(values).__name__ == "DataFrame":
            stations = dtb.index.intersection(values.index)
            labels = values.columns
            flag_values = True
        elif callable(values):
            stations = dtb.index.intersection(file.listing(subdir, "filename"))
            labels = values(station if infer is None else stations[0]).columns
            flag_values = False
        else:
            utils.printing("clusters:", station, "invalid data", level=level)
            raise

        if type(lengths).__name__ in ["Series", "DataFrame"]:
            lengths = lengths.squeeze()
            flag_lengths = True
        elif lengths in dtb.columns:
            lengths = dtb.loc[:, lengths]
            flag_lengths = True
        else:
            lengths = None
            flag_lengths = False

        if heterogeneity is not None:
            if durations is None:
                durations = [
                    label
                    for label in labels
                    if label not in ["index", indexcode, indexdatetime]
                ]
            elif isinstance(durations, (str, int, float)):
                durations = [durations]
            if len(durations) == 1:
                columns = ["H"]
            else:
                columns = ["H %s" % (duration) for duration in durations]
            flag_heterogeneity = True
        else:
            flag_heterogeneity = False

        try:

            if target is None:
                target = dtb.loc[[station], [x, y]]
            elif type(target).__name__ == "DataFrame":
                target = target.loc[[station], [x, y]]
            else:
                target = pandas.DataFrame(
                    data=numpy.array(target).reshape(1, 2),
                    index=pandas.Index([station], name="code"),
                    columns=[x, y],
                )

            # resetting index
            target.reset_index(inplace=True)

            if infer:
                subvalues = pandas.DataFrame()
            elif flag_values:
                subvalues = values.loc[station]
            else:
                subvalues = values(station)

            if (
                infer is None
                and years is not None
                and (
                    pandas.to_datetime(subvalues.loc[station, indexdatetime])
                    .year.unique()
                    .size
                )
                < years
            ):
                utils.printing(
                    "clusters:", station, "scarce years", level=level
                )
                raise

            # array dataframe
            array = dtb.loc[
                stations.difference([station]), [x, y]
            ].reset_index()

            if projections is None:
                import geopy.distance

                array.loc[:, "distance"] = array.apply(
                    lambda i: geopy.distance.geodesic(
                        target[[y, x]].squeeze(), i[[y, x]]
                    ).kilometers,
                    axis=1,
                )
            else:
                array.loc[:, "distance"] = (
                    numpy.add.reduce(
                        (target[[y, x]].squeeze() - array.loc[:, [y, x]]) ** 2,
                        axis=1,
                    )
                    ** 0.5
                )

            # distance sorting
            array.sort_values(by="distance", inplace=True, ignore_index=True)

            # array reindex
            array.index = array.index + 1

            # types dictionary
            types = {
                "array": str,
                x: float,
                y: float,
                "distance": float,
                "length": int,
            }
            if "columns" in locals():
                types.update({i: float for i in columns})

            # initialize entry dataframe
            entry = pandas.DataFrame(
                {i: pandas.Series(dtype=j) for i, j in types.items()}
            )

            if flag_heterogeneity:
                H = functools.partial(
                    region.grandjury,
                    columns=durations,
                    maxima=maxima,
                    index=indexdatetime,
                    order=order,
                    partial=partial,
                    distribution=distribution,
                    arguments=arguments,
                    keywords=keywords,
                    realizations=realizations,
                    dataframe=None,
                    level=1,
                )

            # initialize group list
            group = []

            # initialize subentries dataframe
            subentries = pandas.DataFrame(
                {i: pandas.Series(dtype=j) for i, j in types.items()}
            ).set_index("array")

            # initialize subvalues dataframe
            subvalues = pandas.DataFrame(
                data={
                    indexdatetime: pandas.Series([], dtype="datetime64[ns]")
                },
                index=pandas.Index([], name=indexcode),
            )

            if infer:
                        
                for pointer, code in utils.iterables(
                    array.loc[:, "code"], index=1
                ):

                    # composing cluster
                    cluster = array.loc[range(1, pointer + 1)]

                    if pack or flag_heterogeneity or not flag_lengths:
                        if flag_values:
                            subvalues = pandas.concat(
                                [subvalues, values.loc[[code]]]
                            )
                        else:
                            subvalues = pandas.concat(
                                [
                                    subvalues,
                                    values(code),
                                ]
                            )

                    if (
                        radius is not None
                        and cluster.loc[pointer, "distance"] > radius
                    ):
                        utils.printing(
                            "clusters:", station, "radius limit", level=level
                        )
                        if (
                            pointer == 1
                            or dimension is not None
                            or cumulative is not None
                        ):
                            raise
                        else:
                            flag_radius = False
                    else:
                        flag_radius = True

                    if flag_radius:

                        # updating entry
                        entry.loc[
                            pointer,
                            [
                                i if i != "code" else "array"
                                for i in array.columns
                            ],
                        ] = array.rename(columns={"code": "array"}).loc[
                            pointer
                        ]

                        # updating entry
                        entry.loc[
                            pointer,
                            [
                                i if i != "code" else "array"
                                for i in array.columns
                            ],
                        ] = array.rename(columns={"code": "array"}).loc[
                            pointer
                        ]

                        if flag_lengths:
                            entry.loc[pointer, "length"] = lengths[code]
                        else:
                            entry.loc[pointer, "length"] = (
                                pandas.DatetimeIndex(
                                    subvalues.loc[code, "datetime"]
                                )
                                .year.unique()
                                .size
                            )

                        # updating group
                        group.append(pointer)

                        if flag_heterogeneity:

                            if dimension is None:
                                subgroups = [
                                    list(i)
                                    for i in itertools.combinations(
                                        group, len(group)
                                    )
                                ]
                            elif len(group) >= dimension:
                                subgroups = [
                                    list(i)
                                    for i in itertools.combinations(
                                        group, dimension
                                    )
                                ]
                            else:
                                continue

                            for subgroup in subgroups:
                                if (
                                    utils.concatenating(subgroup)
                                    not in subentries.index
                                ):
                                    # updating subentry
                                    subentries.loc[
                                        utils.concatenating(subgroup),
                                        [x, y, "distance"],
                                    ] = cluster.loc[
                                        subgroup, [x, y, "distance"]
                                    ].mean(
                                        numeric_only=True
                                    )
                                    subentries.loc[
                                        utils.concatenating(subgroup), "length"
                                    ] = entry.loc[subgroup, "length"].sum()

                                    # evaluating heterogeneity
                                    subentries.loc[
                                        utils.concatenating(subgroup), columns
                                    ] = H(
                                        subvalues,
                                        cluster.loc[pandas.Index(subgroup), :],
                                    )

                            if len(columns) != 1:
                                subentries.loc[:, "H"] = subentries.loc[
                                    :, columns
                                ].mean(axis=1)

                            # flagging potential cluster(s)
                            subentries.loc[:, "flag"] = (
                                subentries.loc[:, "H"] <= heterogeneity
                            )

                            if cumulative is not None:
                                subentries.loc[
                                    subentries.loc[:, "flag"], "flag"
                                ] = (
                                    subentries.loc[
                                        subentries.loc[:, "flag"], "length"
                                    ]
                                    >= cumulative
                                )

                            if not ranking:

                                if not any(subentries.loc[:, "flag"]):
                                    utils.printing(
                                        "clusters:",
                                        station,
                                        "keep searching for homogeneous cluster(s)",
                                        level=level,
                                    )
                                    continue
                                else:
                                    # defining group
                                    group = [
                                        int(i)
                                        for i in subentries.loc[
                                            subentries.loc[:, "flag"],
                                            "distance",
                                        ]
                                        .idxmin()
                                        .split()
                                    ]
                                    # flagging entry
                                    entry.loc[:, "flag"] = False
                                    entry.loc[group, "flag"] = True

                        else:
                            if (
                                dimension is not None
                                and len(group) < dimension
                            ):
                                continue
                            elif (
                                cumulative is not None
                                and entry.loc[:, "length"].sum() < cumulative
                            ):
                                continue
                            elif (
                                radius is not None
                                and entry.loc[pointer, "distance"] < radius
                            ):
                                continue

                    if ranking is not None:

                        if len(subgroups) == 1:
                            for idx in subgroup:
                                entry.loc[idx, "h"] = H(
                                    subvalues,
                                    cluster.loc[
                                        (i for i in subgroup if i != idx), :
                                    ],
                                ).mean()
                        else:
                            for idx in entry.index:
                                entry.loc[idx, ["h"]] = subentries.loc[
                                    ~subentries.index.str.contains(str(idx)),
                                    "H",
                                ]

                    # region values
                    entry.loc[pointer + 1, "array"] = "region"

                    if flag_heterogeneity:
                        entry.loc[
                            pointer + 1, [x, y, "distance", "length", "H"]
                        ] = subentries.loc[
                            utils.concatenating(group),
                            [x, y, "distance", "length", "H"],
                        ]
                        if len(columns) != 1:
                            entry.loc[pointer + 1, columns] = subentries.loc[
                                utils.concatenating(group), columns
                            ]

                    else:
                        entry.loc[pointer + 1, [x, y, "distance"]] = entry.loc[
                            :pointer,
                            [x, y, "distance"],
                        ].mean()
                        entry.loc[pointer + 1, "length"] = (
                            entry.loc[:pointer, "length"].sum(),
                        )

                    break

            else:

                # initialize subentries flag
                subentries = None

                for pointer, code in utils.iterables(
                    array.loc[:, "code"], index=1
                ):

                    # composing subcluster
                    subcluster = pandas.concat([target, array.loc[[pointer]]])

                    if pack or flag_heterogeneity or not flag_lengths:

                        if flag_values:
                            subvalues = pandas.concat(
                                [subvalues, values.loc[[code]]]
                            )
                        else:
                            subvalues = pandas.concat(
                                [
                                    subvalues,
                                    values(code),
                                ]
                            )

                    if (
                        radius is not None
                        and subcluster.loc[pointer, "distance"] > radius
                    ):
                        utils.printing(
                            "clusters:", station, "radius limit", level=level
                        )
                        if (
                            pointer == 1
                            or dimension is not None
                            or cumulative is not None
                        ):
                            raise
                        else:
                            flag_radius = False
                    else:
                        flag_radius = True

                    if flag_radius:

                        # updating entry
                        entry.loc[
                            pointer,
                            [
                                i if i != "code" else "array"
                                for i in array.columns
                            ],
                        ] = array.rename(columns={"code": "array"}).loc[
                            pointer
                        ]

                        if flag_lengths:
                            entry.loc[pointer, "length"] = lengths.loc[code][0]
                        else:
                            entry.loc[pointer, "length"] = (
                                pandas.DatetimeIndex(
                                    subvalues.loc[code, "datetime"]
                                )
                                .year.unique()
                                .size
                            )

                        if flag_heterogeneity:
                            # target-pointer heterogeneity
                            entry.loc[pointer, columns] = H(
                                subvalues, subcluster
                            )

                            if len(columns) != 1:
                                entry.loc[pointer, "H"] = (
                                    entry.loc[pointer, columns].dropna().mean()
                                )

                            # flagging potential subcluster
                            entry.loc[pointer, "flag"] = (
                                entry.loc[pointer, "H"] <= heterogeneity
                            )

                            if entry.loc[pointer, "flag"]:
                                utils.printing(
                                    "clusters:",
                                    station,
                                    "homogeneous site",
                                    entry.loc[pointer, "array"],
                                    "added",
                                    level=level,
                                )
                            elif not ranking:
                                utils.printing(
                                    "clusters:",
                                    station,
                                    "heterogeneous site",
                                    entry.loc[pointer, "array"],
                                    "discarderd",
                                    level=level,
                                )
                                continue

                        # updating group
                        group.append(pointer)

                        if dimension is not None and len(group) < dimension:
                            continue
                        elif (
                            cumulative is not None
                            and entry.loc[group, "length"].sum() < cumulative
                        ):
                            continue

                        # composing cluster
                        cluster = pandas.concat([target, array.loc[group]])

                        if flag_heterogeneity:
                            # region heterogeneity including target
                            entry.loc[pointer + 1, columns] = H(
                                subvalues, cluster
                            )
                            if len(columns) != 1:
                                entry.loc[pointer + 1, "H"] = (
                                    entry.loc[pointer + 1, columns]
                                    .dropna()
                                    .mean()
                                )

                            # flagging potential cluster
                            entry.loc[pointer + 1, "flag"] = (
                                entry.loc[pointer + 1, "H"] <= heterogeneity
                            )

                            if ranking is not None:
                                for idx in group:
                                    entry.loc[idx, "h"] = H(
                                        subvalues,
                                        cluster.loc[
                                            (i for i in group if i != idx), :
                                        ],
                                    ).mean()

                            elif not entry.loc[pointer + 1, "flag"]:
                                utils.printing(
                                    "clusters:",
                                    station,
                                    "keep searching for homogeneous cluster",
                                    level=level,
                                )
                                drop = (
                                    entry.loc[group, columns]
                                    .dropna()
                                    .mean(axis=1)
                                    .idxmax()
                                )
                                entry.loc[drop, "flag"] = False
                                group.remove(drop)
                                continue

                    # region values
                    entry.loc[pointer + 1, "array"] = "region"
                    entry.loc[pointer + 1, [x, y, "distance"]] = cluster.loc[
                        group, [x, y, "distance"]
                    ].mean(numeric_only=True)
                    entry.loc[pointer + 1, ["length"]] = entry.loc[
                        group, "length"
                    ].sum()

                    break

            utils.printing(
                "clusters:",
                station,
                "homogeneous cluster found",
                level=level,
            )

            # adjustments
            entry = entry.astype(types).set_index(
                pandas.Index([station] * entry.shape[0], name="code")
            )

            if pack is None:
                output = entry
                outdic = {"index": False, **procdic}
            else:
                output = {
                    "cluster": entry,
                    "subclusters": subentries,
                    "data": subvalues,
                }
                outdic = procdic

            if procdir:
                utils.save(
                    output,
                    station,
                    procdir,
                    procext,
                    proczip,
                    level=1,
                    **outdic,
                )

        except:
            utils.printing(
                "clusters:", station, "processing failed", level=level
            )
            return

        else:
            utils.printing(
                "clusters:", station, "processing success", level=level
            )
            return output


def blockranger(
    values,
    station,
    duration,
    index="datetime",
    blocks=None,
    maxima=None,
    threshold=None,
    length=3,
    delta=None,
    normalize=None,
    function=stats.lm,
    outputs=["L1", "L2", "L3"],
    arguments=(),
    keywords={},
    statistics=True,
    subdir=None,
    subext="csv",
    subzip="gzip",
    subdic={},
    level=0,
):
    """blocks evaluation function"""

    try:
        if isinstance(values, (str, list, tuple)) and subdir is None:
            subdir = values
        if isinstance(subdir, (str, list, tuple)):
            values = region.datafetcher(
                None,
                station,
                subdir=subdir,
                subext=subext,
                subzip=subzip,
                **subdic,
                level=1,
            )

        # extracting data
        data = values.loc[station]

        # entry creation
        entry = stats.blocks(
            data,
            index,
            duration,
            blocks,
            maxima,
            threshold,
            normalize,
            length,
            delta,
            function,
            outputs,
            arguments,
            keywords,
            statistics,
        )

        # setting columns
        entry.loc[:, ["code", "duration"]] = station, duration

    except:
        utils.printing(
            "blockranger:",
            station,
            duration,
            "processing failed",
            level=level,
        )
        return

    else:
        utils.printing(
            "blockranger:",
            station,
            duration,
            "processing success",
            level=level,
        )
        return entry


def regionalization(
    clusters,
    moments,
    station,
    duration,
    infer=None,
    blocks=None,
    orders=["L1", "L2", "L3"],
    coefficients=None,
    level=0,
):
    """moments regionalization function"""

    try:
        # array definition
        array = moments.index.get_level_values("code").intersection(
            clusters.loc[station, "array"].to_numpy()
        )

        if infer is None:
            array = array.union([station])

        if blocks is None:
            # entry creation
            entry = stats.swm(
                moments,
                (array, duration),
                orders,
                scale=None,
                size="λ",
                coefficients=coefficients,
                dataframe=True,
            )

            if numpy.any(numpy.isnan(entry.loc[:, orders])):
                raise

            # entry update
            entry.loc[:, ["δ", "τ", "κ", "μ", "λ"]] = moments.loc[
                (station, duration), ["δ", "τ", "κ", "μ", "λ"]
            ].to_numpy()

            # entry indexing
            entry.loc[:, ["code", "duration", "block", "include"]] = (
                station,
                duration,
                moments.loc[(station, duration), :].index.get_level_values(
                    "block"
                )[0],
                None,
            )

            # entry averages
            entry.loc[:, ["Δ", "T", "K", "M", "Λ"]] = (
                moments.loc[(array, duration), ["δ", "τ", "κ", "μ", "λ"]]
                .mean()
                .to_numpy()
                .squeeze()
            )

        else:
            try:
                blocks_site = (
                    moments.loc[(station, duration), :]
                    .index.get_level_values("block")
                    .unique()
                )
            except:
                blocks_site = pandas.Index([], name="block")

            blocks_array = (
                moments.loc[(array, duration), :]
                .index.get_level_values("block")
                .unique()
                .sort_values()
            )

            # initialize
            entries = []

            for block in blocks_array:
                # entry creation
                entry = stats.swm(
                    moments,
                    (array, duration, block),
                    scale=None,
                    size="λ",
                    coefficients=coefficients,
                    dataframe=True,
                )

                if numpy.any(numpy.isnan(entry.loc[:, orders])):
                    raise

                if block in blocks_site:
                    entry.loc[:, ["δ", "τ", "κ", "μ", "λ"]] = (
                        moments.loc[
                            (station, duration, block),
                            ["δ", "τ", "κ", "μ", "λ"],
                        ]
                        .to_numpy()
                        .squeeze()
                    )

                else:
                    entry.loc[:, ["δ", "τ", "μ", "κ", "λ"]] = numpy.nan

                # setting columns
                entry.loc[:, ["code", "duration", "block", "include"]] = (
                    station,
                    duration,
                    block,
                    block in blocks_site,
                )

                # entry append
                entries.append(entry)

            # entries concatenation
            entry = pandas.concat(entries)

            for i, j in zip(
                [
                    moments.loc[(array, duration), l].dropna()
                    for l in ["δ", "τ", "μ", "κ", "λ"]
                ],
                ["Δ", "T", "M", "K", "Λ"],
            ):
                if infer is None:
                    # station value
                    i_station = i.loc[station].mean()
                    # block spatial average value
                    i_block = i.groupby("block").mean().dropna().to_numpy()
                    # block temporal ratio value
                    i_ratio = i_block / i_block.mean()
                    entry.loc[:, [j]] = i_station * i_ratio
                else:
                    entry.loc[:, [j]] = (
                        i.groupby("block").mean().dropna().to_numpy()
                    )

    except:
        utils.printing(
            "regionalization:",
            station,
            duration,
            "processing failed",
            level=level,
        )
        return

    else:
        utils.printing(
            "regionalization:",
            station,
            duration,
            "processing success",
            level=level,
        )
        return entry


def empiricals(
    values,
    station,
    duration,
    normalize=None,
    position="weibull",
    subdir=None,
    subext="csv",
    subzip="gzip",
    subdic={},
    level=0,
):
    """empiricals evaluation function"""

    try:
        if subdir is None:
            subdir = values

        if isinstance(values, (str, list, tuple)) and subdir is None:
            subdir = values
        if isinstance(subdir, (str, list, tuple)):
            data = region.datafetcher(
                None,
                station,
                subdir=subdir,
                subext=subext,
                subzip=subzip,
                **subdic,
                level=1,
            )
        elif values is not None:
            data = values.loc[station]

        # data values
        data = data.set_index(
            pandas.DatetimeIndex(data.loc[:, "datetime"])
        ).loc[:, duration]

        # annual maxima
        annual = data.resample("y").max().dropna()

        # initialize
        index = []

        for datetime, value in annual.items():
            subdata = data.loc[data.index.year == datetime.year]
            index.append(subdata.index[numpy.where(subdata == value)[0][0]])

        # annual indexes
        index = pandas.DatetimeIndex(index)

        if normalize == "dataset":
            annual = annual * data.mean()
        if normalize == "maxima":
            annual = annual * annual.mean()

        # entry creation
        entry = dists.empirical(annual, position)

        # setting columns
        entry.loc[:, "datetime"] = index
        entry.loc[:, "block"] = index.year
        entry.loc[:, ["code", "duration"]] = station, duration

    except:
        utils.printing(
            "empiricals:",
            station,
            duration,
            "processing failed",
            level=level,
        )
        return

    else:
        utils.printing(
            "empiricals:",
            station,
            duration,
            "processing success",
            level=level,
        )
        return entry


def quantiles(
    values,
    parameters,
    station,
    duration,
    clusters=None,
    infer=None,
    blocks=None,
    normalize=True,
    rescale=True,
    empirical=True,
    position="weibull",
    estimate="x",
    curve=None,
    metastatistics=None,
    coefficients=None,
    arguments=(),
    dictionary={},
    distribution="gev",
    labels=["shape", "scale", "location"],
    flags="flag",
    subdir=None,
    subext="csv",
    subzip="gzip",
    subdic={},
    level=0,
):
    """quantiles evaluation function"""

    try:

        # projections
        n, par = region.projector(
            parameters, station, duration, "n", labels, clusters, infer, blocks
        )

        if infer is None:
            τ = parameters.loc[(station, duration), "τ"].mean()
            κ = parameters.loc[(station, duration), "κ"].mean()
        else:
            τ = parameters.loc[
                (
                    clusters.loc[
                        ~clusters.loc[:, "array"].isin(
                            ["site", "envelope", "region"]
                        )
                    ].loc[station, "array"],
                    duration,
                ),
                "τ",
            ].mean()
            κ = parameters.loc[
                (
                    clusters.loc[
                        ~clusters.loc[:, "array"].isin(
                            ["site", "envelope", "region"]
                        )
                    ].loc[station, "array"],
                    duration,
                ),
                "κ",
            ].mean()

        if coefficients:
            coefficients = (
                clusters.loc[
                    ~clusters.loc[:, "array"].isin(
                        ["site", "envelope", "region"]
                    )
                ]
                .loc[
                    station,
                    (
                        [coefficients]
                        if isinstance(coefficients, (int, str))
                        else coefficients
                    ),
                ]
                .to_numpy()
            )
            if infer is None:
                coefficients = numpy.concatenate(
                    [
                        numpy.nanmean(coefficients, axis=0, keepdims=True),
                        coefficients,
                    ]
                )

        if empirical:

            if isinstance(values, (str, list, tuple)) and subdir is None:
                subdir = values
            if isinstance(subdir, (str, list, tuple)):
                data = region.datafetcher(
                    None,
                    station,
                    subdir=subdir,
                    subext=subext,
                    subzip=subzip,
                    **subdic,
                    level=1,
                )
            elif "code" in values.index.names:
                data = values.loc[station]
            else:
                data = values

            # data values
            data = data.set_index(
                pandas.DatetimeIndex(data.loc[:, "datetime"])
            ).loc[:, "value" if "value" in data.columns else duration]

            # annual maxima
            annual = data.resample("y").max().dropna()

            # entry creation
            values = dists.empirical(annual, position)

            # initialize
            index = []

            for datetime, value in annual.items():
                # geting data
                subdata = data.loc[data.index.year == datetime.year]

                # getting index
                index.append(
                    subdata.index[numpy.where(subdata == value)[0][0]]
                )

            # annual indexes
            index = pandas.DatetimeIndex(index)

            # values indexing
            values.loc[:, "datetime"] = index
            values.loc[:, "block"] = index.year

        elif type(values).__name__ != "DataFrame":
            if estimate == "x":
                values = pandas.DataFrame(data=values, columns=["F"])
            if estimate == "F":
                values = pandas.DataFrame(data=values, columns=["value"])

        elif "code" in values.index.names:
            values = values.loc[station]

        if normalize is None:
            k = 1
        elif normalize is True:
            k = κ
        elif isinstance(normalize, (int, float)):
            k = 1 / normalize
        elif normalize == "maxima" and empirical:
            k = 1 / annual.mean()
        elif normalize == "dataset" and empirical:
            k = 1 / data.mean()
        else:
            utils.printing(
                "quantiles:",
                station,
                duration,
                "invalid normalize",
                level=level,
            )
            raise

        if rescale is None:
            r = 1
        elif rescale is True:
            r = 1 / κ
        elif isinstance(rescale, (int, float)):
            r = 1 / rescale
        else:
            utils.printing(
                "quantiles:",
                station,
                duration,
                "invalid rescale",
                level=level,
            )
            raise

        if "value" in values.columns:
            values.loc[:, "value"] = values.loc[:, "value"].mul(k)

        if metastatistics is None:
            x = dists.quantile(
                values.loc[:, "F"],
                distribution,
                par,
                *arguments,
                c_x="model",
                c_F="P",
                threshold=τ * k,
                curve=None,
                **dictionary,
            )
            if curve is not None:
                curve = dists.quantile(
                    None,
                    distribution,
                    par,
                    *arguments,
                    c_x="model",
                    c_F="P",
                    threshold=τ * k,
                    curve=curve,
                    **dictionary,
                )
            if estimate == "F":
                F = values.assign(
                    P=dists.cdf(
                        values.loc[:, "value"],
                        distribution,
                        par,
                        *arguments,
                        threshold=τ * k,
                        reshape=1,
                        **dictionary,
                    ).squeeze()
                )

        else:
            x = mev.quantile(
                values.loc[:, "F"],
                distribution,
                par,
                n,
                *arguments,
                **(
                    {
                        "v_0": (
                            annual.min() * k
                            if empirical is not None
                            else τ * k
                        )
                    }
                    if "v_0" not in dictionary.keys()
                    and (empirical is not None or τ != 0)
                    else {}
                ),
                coefficients=coefficients,
                c_x="model",
                c_F="P",
                threshold=τ * k,
                curve=None,
                **dictionary,
            )
            if curve is not None:
                curve = mev.quantile(
                    None,
                    distribution,
                    par,
                    n,
                    *arguments,
                    **(
                        {
                            "v_0": (
                                annual.min() * k
                                if empirical is not None
                                else τ * k
                            )
                        }
                        if "v_0" not in dictionary.keys()
                        and (empirical is not None or τ != 0)
                        else {}
                    ),
                    c_x="model",
                    c_F="P",
                    threshold=τ * k,
                    curve=curve,
                    **dictionary,
                )
            if estimate == "F":
                F = values.assign(
                    P=mev.spatiotemporal(
                        values.loc[:, "value"],
                        distribution,
                        par,
                        n,
                        *arguments,
                        coefficients=coefficients,
                        threshold=τ * k,
                        **dictionary,
                    )
                )

        if "x" in locals():

            if (
                flags in x.columns
                and x.loc[:, flags].sum() != x.loc[:, flags].count()
            ):
                raise

        if estimate == "x":
            # initialize entry
            entry = dists.evaluation(
                values,
                x,
                relate=["F", "P"],
                source="model",
                destination="model",
            )
            if curve is not None:
                entry = pandas.concat([entry, curve]).reset_index(drop=True)
                entry.loc[0 : values.size, "P"] = values.loc[:, "F"]
                entry.sort_values("P", inplace=True)

        elif estimate == "F":
            # initialize entry
            entry = F
            if curve is not None:
                entry = pandas.concat([entry, curve]).reset_index(drop=True)
                entry.loc[0 : F.size, "model"] = F.loc[:, "value"]
                entry.sort_values("P", inplace=True)

        # setting columns
        entry.loc[:, ["code", "duration"]] = station, duration

        if "value" in entry.columns:
            entry.loc[:, "value"] = entry.loc[:, "value"].mul(r)
        if "model" in entry.columns:
            entry.loc[:, "model"] = entry.loc[:, "model"].mul(r)

    except:
        utils.printing(
            "quantiles:",
            station,
            duration,
            "processing failed",
            level=level,
        )
        return

    else:
        utils.printing(
            "quantiles:",
            station,
            duration,
            "processing success",
            level=level,
        )
        return entry


def jocker(
    values,
    station=None,
    duration=None,
    benchmark=None,
    clusters=None,
    flags="flag",
    normalize=None,
    threshold=None,
    include=None,
    samples=1e3,
    combinations=None,
    overlapping=None,
    proportional=None,
    sizes=None,
    lengths=3,
    coefficients=None,
    distributions=["gev", "mev", "smev"],
    dictionaries=None,
    drop=None,
    focus=None,
    aggregate=None,
    apply=None,
    compact=True,
    pack=None,
    skip=None,
    out=None,
    subdir=None,
    subext="csv",
    subzip=None,
    subdic={},
    procdir="validation",
    proczip=None,
    procdic={},
    level=0,
):
    """cross-validator function"""

    if pack is None:
        procext = "csv"
    else:
        procext = "pkl"

    if benchmark == "site":
        sizes = None

    # indexers
    distributions, sizes, lengths = utils.indexer(
        distributions, sizes, lengths
    )

    if skip and (
        file.flag(
            [station, duration, benchmark],
            procdir,
            procext,
        )
    ):
        if out:
            utils.printing(
                "jocker:",
                station,
                duration,
                benchmark,
                "processing resumed",
                level=level,
            )
            entry = utils.load(
                [station, duration, benchmark],
                procdir,
                procext,
                proczip,
                **procdic,
                level=1,
            )
        else:
            utils.printing(
                "jocker:",
                station,
                duration,
                benchmark,
                "processing skipped",
                level=level,
            )
            return

    else:
        try:
            if isinstance(values, (str, list, tuple)) and subdir is None:
                subdir = values
            if isinstance(subdir, (str, list, tuple)):
                data = datafetcher(
                    station,
                    subdir=subdir,
                    subext=subext,
                    subzip=subzip,
                    **subdic,
                    level=1,
                )
            elif values is not None:
                data = values.loc[station]

            if clusters is not None:
                clusters = clusters.loc[
                    ~clusters.loc[:, "array"].isin(
                        ["site", "envelope", "region"]
                    )
                ].loc[station, :]
                if benchmark != "region":
                    clusters = clusters.loc[
                        clusters.loc[:, "array"] != station
                    ]
                if flags in clusters.columns:
                    clusters = clusters.loc[clusters.loc[:, flags]]

                if isinstance(coefficients, (int, str)):
                    coefficients = [coefficients]

                if coefficients is not None:
                    coefficients = pandas.DataFrame(
                        numpy.hstack(
                            [
                                clusters.loc[:, [i]].to_numpy()
                                for i in coefficients
                            ]
                        ),
                        columns=coefficients,
                        index=clusters.loc[:, "array"],
                    )

            # adjustments
            data = (
                data.set_index(pandas.DatetimeIndex(data.loc[:, "datetime"]))
                .drop(columns="datetime")
                .loc[:, duration]
                .dropna()
            )

            # maxima values
            annual = data.resample("y").max().dropna()

            if threshold is not None:
                τ_data = stats.quantile(data.to_numpy(), threshold)
            else:
                τ_data = 0

            # maxima threshold
            τ_annual = 0

            # applying thresholds
            data = data.loc[data > τ_data] - τ_data
            annual = annual.loc[annual > τ_annual] - τ_annual

            # dictionaries
            keywords = {i: {} for i in distributions}

            if dictionaries is not None:
                keywords.update(dictionaries)

            # station indexers
            idx_site = pandas.Index([station], name="code")

            if benchmark == "site":
                # site indexer
                sizes = utils.indexer(None)

                # indexers iterable
                idxs = pandas.DataFrame(
                    zip([idx_site], [None], [None]),
                    columns=["site", "envelope", "region"],
                ).iterrows()

            else:
                # envelope indexers
                idx_envelope = pandas.Index(
                    clusters.loc[clusters.loc[:, "array"] != station, "array"],
                    name="code",
                )

                if (
                    not pandas.isnull(sizes.max())
                    and sizes.max() > idx_envelope.size
                ):
                    utils.printing(
                        "jocker:",
                        station,
                        duration,
                        benchmark,
                        "invalid cluster size",
                        level=level,
                    )
                    return

                if combinations is None:
                    # envelope indexer
                    idx_envelope = utils.indexer(
                        [
                            (
                                idx_envelope
                                if size is None
                                else idx_envelope[:size]
                            )
                            for size in sizes
                        ]
                    )
                else:
                    # envelope indexer
                    idx_envelope = utils.indexer(
                        [
                            utils.indexer(i)
                            for size in sizes
                            for i in itertools.combinations(
                                idx_envelope, r=size
                            )
                        ]
                    )

                # region indexer
                idx_region = utils.indexer(
                    [
                        idx_site.union(i, sort=False).unique()
                        for i in idx_envelope
                    ]
                )

                # indexers iterable
                idxs = pandas.DataFrame(
                    zip(
                        [idx_site] * idx_region.size, idx_envelope, idx_region
                    ),
                    columns=["site", "envelope", "region"],
                ).iterrows()

            # initialize
            entry = []

            for combination, idx, length in utils.iterables(
                idxs, lengths, product=True, index=1
            ):

                # calibration indexer
                idx_calibration = idx[-1][benchmark]

                # size
                S = idx_calibration.size

                if isinstance(subdir, (str, list, tuple)):
                    data_calibration = (
                        region.datafetcher(
                            None,
                            idx_calibration,
                            subdir=subdir,
                            subext=subext,
                            subzip=subzip,
                            **subdic,
                            level=1,
                        )
                        .loc[:, ["datetime", duration]]
                        .dropna()
                    )
                elif values is not None:
                    data_calibration = values.loc[
                        idx_calibration, ["datetime", duration]
                    ].dropna()

                if benchmark != "envelope":
                    κ_annual = 1 / annual.mean()
                else:
                    κ_annual = (
                        region.blockranger(
                            None,
                            data_calibration,
                            idx_calibration,
                            duration,
                            blocks=None,
                            maxima="y",
                            threshold=threshold,
                            normalize="maxima",
                            filename=None,
                            level=1,
                        )
                        .loc[:, "κ"]
                        .mean()
                    )

                if "mev" in distributions:
                    # computing moments
                    dataframe_moments = region.blockranger(
                        None,
                        data_calibration,
                        idx_calibration,
                        duration,
                        blocks="y",
                        threshold=threshold,
                        normalize="dataset",
                        filename=None,
                        level=1,
                    )
                    # computing projections
                    array_moments = region.projector(
                        dataframe_moments,
                        station,
                        duration,
                        size="λ",
                        clusters=idx_calibration,
                        infer=True,
                        blocks=True,
                    )

                if benchmark != "envelope":
                    κ_data = 1 / data.mean()
                else:
                    κ_data = dataframe_moments.loc[:, "κ"].mean()

                if coefficients is None:
                    array_coefficients = None
                else:
                    array_coefficients = coefficients.loc[
                        idx_calibration
                    ].to_numpy()

                # defining blocks
                blocks_validation = pandas.DataFrame(
                    data=annual.index.year.to_numpy(dtype=int),
                    index=pandas.Index(
                        [station] * annual.index.year.to_numpy(dtype=int).size,
                        name="code",
                    ),
                    columns=["block"],
                    dtype=int,
                )
                if "dataframe_moments" in locals():
                    blocks_calibration = (
                        dataframe_moments.reset_index()
                        .set_index("code")
                        .loc[:, ["block"]]
                        .astype(int)
                    )
                else:
                    blocks_calibration = (
                        region.blockranger(
                            None,
                            data_calibration,
                            idx_calibration,
                            blocks="y",
                            threshold=threshold,
                            statistics=False,
                            filename=None,
                            level=1,
                        )
                        .reset_index()
                        .set_index("code")
                        .loc[:, ["block"]]
                        .astype(int)
                    )

                if benchmark != "site":
                    blocks_region = pandas.concat(
                        [blocks_validation, blocks_calibration]
                    )
                    matrix_region = validation.matrix(
                        idx[-1]["region"], blocks_region
                    )
                    idx_filter = utils.indexer(
                        [
                            i[0]
                            for i in enumerate(matrix_region[:, :, 1:])
                            if ~numpy.isnan(i[1]).all()
                        ]
                    )

                    if overlapping:
                        matrix_region = (
                            matrix_region[
                                idx_filter,
                                :,
                                :,
                            ],
                        )

                    if benchmark == "envelope":
                        matrix_validation, matrix_calibration = (
                            matrix_region[:, :, :1],
                            matrix_region[idx_filter, :, 1:],
                        )

                    if benchmark == "region":
                        matrix_validation, matrix_calibration = (
                            matrix_region[:, :, :1],
                            matrix_region[:, :, 0:],
                        )

                else:
                    matrix_validation, matrix_calibration = validation.matrix(
                        idx[-1]["site"], blocks_validation
                    ), validation.matrix(idx_calibration, blocks_calibration)
                    idx_filter = utils.indexer(
                        list(range(matrix_validation.size))
                    )

                if isinstance(length, int):
                    r = annual.size / length
                else:
                    if isinstance(length, str) and ("." in length):
                        r = float(length)
                    elif isinstance(length, str) and ("%" in length):
                        r = (
                            float(
                                int("".join(i for i in length if i.isdigit()))
                            )
                            * 1e-2
                        )
                    length = int(r * annual.size)

                if not isinstance(samples, int):
                    samples = int(samples)
                if (
                    samples
                    > math.comb(
                        numpy.min(
                            numpy.sum(~numpy.isnan(matrix_calibration), axis=0)
                        ),
                        length,
                    )
                    ** S
                ):
                    utils.printing(
                        "jocker:",
                        station,
                        duration,
                        benchmark,
                        "invalid samples length",
                        level=level,
                    )
                    continue

                if include is None:
                    indexer_validation = validation.shuffler(
                        matrix_validation[:, 0, :],
                        skip=None,
                        size=length,
                        samples=samples ** (S if proportional else 1),
                        unique="vectors",
                        reshape=True,
                    )
                    V = annual.shape[0] - length
                else:
                    indexer_validation = None
                    V = annual.shape[0]

                # calibrations
                indexer_calibration = validation.shuffler(
                    matrix_calibration[:, 0, :],
                    skip=(
                        None if include else indexer_validation[idx_filter, :]
                    ),
                    size=length,
                    samples=samples ** (S if proportional else 1),
                    unique="arrays",
                    reshape=False,
                )
                C = numpy.sum(~numpy.isnan(indexer_calibration[:, :, 0]))

                if include is None:
                    empiricals = validation.empiricals(
                        annual,
                        indexer_validation,
                        matrix_validation,
                        normalize=None,
                        invert=True,
                        maxima=None,
                    )
                else:
                    empiricals = [dists.empirical(annual)] * samples

                # computing constants
                τ_annual_calibration, κ_annual_calibration = numpy.split(
                    region.blockranger(
                        None,
                        data_calibration,
                        idx_calibration,
                        duration,
                        blocks=None,
                        maxima="y",
                        threshold=None,
                        normalize="subset",
                        filename=None,
                        level=1,
                    )
                    .loc[:, ["τ", "κ"]]
                    .groupby(by=["code", "duration"], sort=None)
                    .mean()
                    .to_numpy(),
                    [1],
                    axis=1,
                )

                if "dataframe_moments" in locals():
                    # computing constants
                    τ_data_calibration, κ_data_calibration = numpy.split(
                        dataframe_moments.loc[:, ["τ", "κ"]]
                        .groupby(by=["code", "duration"], sort=None)
                        .mean()
                        .to_numpy(),
                        [1],
                        axis=1,
                    )
                else:
                    τ_data_calibration, κ_data_calibration = None, None

                # initialize
                quantiles = []

                if "gev" in distributions:
                    quantiles.append(
                        cross.function(
                            "gev",
                            empiricals,
                            data_calibration,
                            indexer_calibration,
                            matrix_calibration,
                            threshold=τ_annual_calibration,
                            normalize=κ_annual_calibration,
                            coefficients=None,
                            τ=τ_annual,
                            κ=κ_annual,
                            **keywords["gev"],
                        )
                    )

                if "mev" in distributions:
                    quantiles.append(
                        cross.function(
                            "mev",
                            empiricals,
                            *array_moments,
                            indexer_calibration,
                            coefficients=None,
                            v=annual.min(),
                            τ=τ_data,
                            κ=κ_data,
                            **keywords["mev"],
                        ),
                    )

                if "smev" in distributions:
                    quantiles.append(
                        cross.function(
                            "smev",
                            empiricals,
                            data_calibration,
                            indexer_calibration,
                            matrix_calibration,
                            threshold=τ_data_calibration,
                            normalize=κ_data_calibration,
                            coefficients=None,
                            v=annual.min(),
                            τ=τ_data,
                            κ=κ_data,
                            **keywords["smev"],
                        )
                    )

                # concatenating
                quantiles = pandas.concat(quantiles)

                if drop == "variable":
                    quantiles.drop(
                        quantiles.loc[
                            quantiles.loc[:, "model"].isna()
                        ].index.unique(),
                        inplace=True,
                    )
                elif drop == "consistent":
                    quantiles.drop(
                        quantiles.loc[quantiles.loc[:, "model"].isna()]
                        .index.get_level_values("extraction")
                        .unique()
                    )

                # appending
                entry.append(
                    quantiles.assign(combination=combination, S=S, V=V, C=C)
                )

            # concatenating
            entry = pandas.concat(entry)

            if normalize:
                entry.loc[:, ["value", "model"]] = entry.loc[
                    :, ["value", "model"]
                ].mul(κ_data)

            if compact:
                entry = frames.reduce(
                    entry,
                    "model",
                    "distribution",
                    ["extraction", "F", "combination"],
                )

            if procdir:
                utils.save(
                    entry,
                    [station, duration, benchmark],
                    procdir,
                    procext,
                    proczip,
                    level=1,
                    **procdic,
                )

        except:
            utils.printing(
                "jocker:",
                station,
                duration,
                benchmark,
                "processing failed",
                level=level,
            )
            return

        else:
            utils.printing(
                "jocker:",
                station,
                duration,
                benchmark,
                "processing success",
                level=level,
            )

    # available distributions
    distributions = [
        i for i in entry.columns if i.partition("_")[0] in distributions
    ]

    if len(distributions) == 0:
        utils.printing(
            "jocker:",
            station,
            duration,
            benchmark,
            "empty output",
            level=level,
        )
        return

    if compact:
        entry = frames.expand(entry, "model", distributions, "distribution")

    # indexing
    entry = entry.assign(
        code=station, duration=duration, benchmark=benchmark
    ).set_index(["code", "duration", "benchmark", "distribution"])

    # filtering
    entry = entry.loc[
        (slice(None), slice(None), slice(None), distributions), :
    ]

    if isinstance(focus, dict):
        entry = frames.focus(entry, **focus)

    if isinstance(aggregate, dict):
        entry = frames.aggregate(entry, **aggregate)

    if isinstance(apply, dict):
        entry = frames.apply(entry, **apply)
    elif apply is not None:
        entry = frames.apply(entry, apply, ["model", "value"], keep=True)

    if out:
        return entry
    else:
        return
