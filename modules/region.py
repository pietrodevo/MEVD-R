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
import numpy
import pandas

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import dists
import file
import sites
import stats
import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def converter(
    filename=None,
    subdir=None,
    procdir=None,
    cpu=1,
    **keywords,
):
    """converter function"""

    if subdir is None and procdir is not None:
        subdir = procdir
    if procdir is None and subdir is not None:
        procdir = subdir

    if filename is None:
        filename = file.listing(procdir, "filename")
    else:
        filename = utils.indexer(filename)

    # defining function
    function = functools.partial(
        sites.converter,
        subdir=subdir,
        procdir=procdir,
        **keywords,
    )

    # processing results
    utils.process(function, filename, cpu=cpu)


def siterunner(
    database,
    function,
    stations=None,
    keys=None,
    index="dataset",
    nested=None,
    dictionaries={},
    out=None,
    **keywords,
):
    """siterunner function"""

    if not callable(function):
        if isinstance(function, str):
            function = utils.definition(function)
        elif isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)

    if stations is None:
        stations = database.index.unique()
    else:
        database = database.loc[stations]

    if keys is None:
        indexers = database.loc[:, index].unique()
    else:
        indexers = database.loc[
            database.loc[:, index].isin(keys), index
        ].unique()

    if out:
        outputs = []

    for indexer in indexers:

        # dataset stations
        stations = database.index[(database.loc[:, index] == indexer)].unique()

        if indexer in dictionaries.keys():
            dictionary = dictionaries[indexer]
        else:
            dictionary = {}

        # dictionary update
        dictionary.update(keywords)

        if nested:
            dictionary.update(
                {
                    "subdir": utils.concatenating(
                        dictionary["subdir"], indexer, separator="/"
                    )
                }
            )

        # function execution
        output = function(database, stations, **dictionary)

        if out:
            outputs.append(output)

    if out and len(outputs) > 0:
        return outputs


def datafetcher(
    dtb=None,
    stations=None,
    index="code",
    outdir="data",
    outdic={},
    subdir="data",
    subdic={},
    filename=None,
    extension="csv",
    compression=None,
    cpu=1,
    **keywords,
):
    """datafetcher function"""

    if stations is None:
        if dtb is None:
            stations = file.listing(subdir, "filename")
        else:
            stations = dtb.index

    # indexers
    stations = utils.indexer(stations)

    if subdir:
        stations = stations.intersection(file.listing(subdir, "filename"))

    # defining function
    function = functools.partial(
        sites.datafetcher,
        subdir=subdir,
        subdic=subdic,
        **keywords,
    )

    # processing results
    results = utils.process(function, stations, cpu=cpu)

    if results is None:
        return

    # dataframe
    dataframe = pandas.concat(results)

    if index is not None:
        dataframe.set_index(index, inplace=True)

    if filename is not None:
        utils.save(
            dataframe, filename, outdir, extension, compression, **outdic
        )

    return dataframe


def stormwalker(
    dtb=None,
    stations=None,
    subdir="series",
    subdic={},
    procdir="extraction",
    procdic={},
    procset=None,
    cpu=1,
    **keywords,
):
    """storm events analyses"""

    if stations is None:
        if dtb is None:
            stations = file.listing(subdir, "filename")
        else:
            stations = dtb.index

    if procset == "difference":
        stations = stations.difference(file.listing(procdir, "filename"))
    elif procset == "intersection":
        stations = stations.intersection(file.listing(procdir, "filename"))

    if dtb is not None and "frequency" in dtb.columns:
        frequencies = dtb.loc[stations, "frequency"]
    elif "frequency" in keywords:
        frequencies = [keywords["frequency"]] * len(stations)
    else:
        frequencies = [None] * len(stations)
    if dtb is not None and "filename" in dtb.columns:
        filenames = dtb.loc[stations, "filename"]
    elif "filename" in keywords:
        filenames = [keywords["filename"]] * len(stations)
    else:
        filenames = [None] * len(stations)
        keywords.pop("filename", None)

    # removing duplicates
    keywords.pop("frequency", None)
    keywords.pop("filename", None)

    # indexers
    stations, frequencies, filenames = utils.indexer(
        stations,
        frequencies,
        filenames,
    )

    # defining function
    function = functools.partial(
        sites.stormwalker,
        subdir=subdir,
        subdic=subdic,
        procdir=procdir,
        procdic=procdic,
        **keywords,
    )

    return utils.process(
        function,
        stations,
        frequencies,
        filenames,
        product=False,
        cpu=cpu,
    )


def grandjury(
    values,
    stations,
    columns=None,
    maxima=None,
    index="datetime",
    order="L2",
    partial=None,
    distribution="gev",
    arguments=(),
    keywords={},
    realizations=1e3,
    dataframe=True,
    subdir=None,
    subext="csv",
    subzip="gzip",
    subdic={},
    level=0,
):
    """clusters homogeinity metric"""

    if type(stations).__name__ == "DataFrame":
        stations = stations.loc[:, "code"]

    if columns is None:
        columns = [column for column in values.columns if column != index]
    elif not isinstance(columns, list):
        columns = [columns]

    if isinstance(values, (str, list, tuple)) and subdir is None:
        subdir = values
    if isinstance(subdir, (str, list, tuple)):
        values = datafetcher(
            None,
            stations,
            subdir=subdir,
            subext=subext,
            subzip=subzip,
            subdic=subdic,
            level=1,
        )

    if isinstance(order, int):
        # indexing position
        position = order - 1

        # formatting label
        order = "L" + str(order)
    else:
        # indexing position
        position = int("".join(i for i in order if i.isdigit())) - 1

    # initilize
    H = numpy.full((1, len(columns)), numpy.nan)

    if maxima:
        normalize = "maxima"
    else:
        normalize = "dataset"

    for i, column in enumerate(columns):

        # array moments
        mom = blockranger(
            None,
            values,
            stations,
            column,
            blocks=None,
            maxima=maxima,
            index=index,
            normalize=normalize,
            filename=None,
            level=1,
        )

        if mom.index.get_level_values("code").difference(stations).size != 0:
            if partial:
                continue
            else:
                break

        # scaled weighted moments
        swm = stats.swm(mom, stations, scale=None, dataframe=False)

        # parameters estimation
        par = dists.parameters(
            distribution, *arguments, array=swm, **keywords
        ).squeeze()

        # sizes
        n = mom.loc[:, "λ"].to_numpy(dtype=int)

        # array coefficients
        t_a = mom.loc[:, order].to_numpy()
        T_a = swm[position]

        # realizations
        realizations = int(realizations)

        # data generation
        data = [
            dists.quantile(
                numpy.random.rand(n[i], realizations),
                distribution,
                par,
                *arguments,
                **keywords,
                dataframe=False,
            ).reshape((-1, realizations))
            for i in range(n.size)
        ]

        # realizations coefficients
        t_r = numpy.vstack(
            [
                numpy.hstack(
                    [
                        stats.lm(i[:, r], order=position + 1)[:, position]
                        for i in data
                    ]
                )
                for r in range(realizations)
            ]
        )
        T_r = (
            numpy.add.reduce(t_r * numpy.tile(n, (realizations, 1)), axis=1)
            / numpy.add.reduce(numpy.tile(n, (realizations, 1)), axis=1)
        ).reshape((realizations, 1))

        # computing statistics
        V_a = numpy.sqrt(
            numpy.add.reduce(n * (t_a - T_a) ** 2, axis=0)
            / numpy.add.reduce(n, axis=0)
        )
        V_r = numpy.sqrt(
            numpy.add.reduce(n * (t_r - T_r) ** 2, axis=1)
            / numpy.add.reduce(n, axis=0)
        )

        # heterogeneity measure
        H[:, i] = (V_a - numpy.mean(V_r)) / numpy.std(V_r)

    if dataframe:
        return pandas.DataFrame(index=[stations[0]], columns=columns, data=H)
    else:
        return H


def clusters(
    dtb,
    values,
    stations=None,
    target=None,
    outdir="data",
    filename="clusters",
    extension="csv",
    compression=None,
    subdir=None,
    subdic={},
    procdir="clusters",
    procdic={},
    procset=None,
    cpu=1,
    **keywords,
):
    """clusters definition function"""

    if stations is None:
        if target is None:
            stations = dtb.index
        else:
            if type(target).__name__ == "DataFrame":
                stations = target.index
            else:
                stations = target

    # indexers
    stations = utils.indexer(stations)

    if isinstance(values, (str, list, tuple)) and subdir is None:
        subdir = values
    if isinstance(subdir, (str, list, tuple)):
        dtb = dtb.loc[dtb.index.intersection(file.listing(subdir, "filename"))]

        if target is None:
            stations = stations.intersection(file.listing(subdir, "filename"))

    if procset == "difference":
        stations = stations.difference(file.listing(procdir, "filename"))
    elif procset == "intersection":
        stations = stations.intersection(file.listing(procdir, "filename"))

    # defining function
    function = functools.partial(
        sites.clusters,
        dtb,
        values,
        target=target,
        subdir=subdir,
        subdic=subdic,
        procdir=procdir,
        procdic=procdic,
        **keywords,
    )

    # processing results
    results = utils.process(function, stations, cpu=cpu)

    if results is None:
        return

    # dataframe
    dataframe = pandas.concat(results)

    if filename is not None:
        utils.save(dataframe, filename, outdir, extension, compression)

    return dataframe


def blockranger(
    dtb,
    values,
    stations=None,
    durations=["01m", "01h", "01d"],
    outdir="data",
    filename="blocks",
    extension="csv",
    compression=None,
    cpu=1,
    **keywords,
):
    """blocks evaluation function"""

    if stations is None:
        if type(dtb).__name__ in ["Series", "DataFrame"]:
            stations = dtb.index
        elif isinstance(dtb, str):
            stations = file.listing(dtb, output="filename")
        elif isinstance(dtb, (list, tuple)):
            stations = file.listing(*dtb, output="filename")
        elif isinstance(dtb, (dict)):
            stations = file.listing(**dtb, output="filename")
        else:
            raise

    # indexers
    stations, durations = utils.indexer(stations, durations)

    if isinstance(values, (str, list, tuple)):
        stations = stations.intersection(file.listing(values, "filename"))
    else:
        stations = stations.intersection(values.index)

    # defining function
    function = functools.partial(sites.blockranger, values, **keywords)

    # processing results
    results = utils.process(function, stations, durations, cpu=cpu)

    if results is None:
        return

    # dataframe
    dataframe = (
        pandas.concat(results)
        .dropna()
        .reset_index()
        .set_index(["code", "duration", "block"])
    )

    if filename is not None:
        utils.save(dataframe, filename, outdir, extension, compression)

    return dataframe


def regionalization(
    dtb,
    clusters,
    moments,
    stations=None,
    durations=["01m", "01h", "01d"],
    flags="flag",
    outdir="data",
    filename="rmoments",
    extension="csv",
    compression=None,
    cpu=1,
    **keywords,
):
    """moments regionalization function"""

    if stations is None:
        if type(dtb).__name__ in ["Series", "DataFrame"]:
            stations = dtb.index
        elif isinstance(dtb, str):
            stations = file.listing(dtb, output="filename")
        elif isinstance(dtb, (list, tuple)):
            stations = file.listing(*dtb, output="filename")
        elif isinstance(dtb, (dict)):
            stations = file.listing(**dtb, output="filename")
        else:
            raise

    # clusters coherency
    clusters = clusters.loc[clusters.loc[:, "array"] != "region"]

    if flags in clusters.columns:
        clusters = clusters.loc[clusters.loc[:, flags]]

    # reindexing
    moments = moments.reset_index().set_index(["code", "duration", "block"])

    # indexers
    stations, durations = utils.indexer(stations, durations)

    # defining function
    function = functools.partial(
        sites.regionalization, clusters, moments, **keywords
    )

    # processing results
    results = utils.process(function, stations, durations, cpu=cpu)

    if results is None:
        return

    # dataframe
    dataframe = pandas.concat(results).set_index(["code", "duration", "block"])

    if filename is not None:
        utils.save(dataframe, filename, outdir, extension, compression)

    return dataframe


def parametersurveyor(
    moments,
    distribution="gev",
    arguments=(),
    keywords={},
    labels=["shape", "scale", "location"],
    orders=["L1", "L2", "L3"],
    regional=None,
    outdir="data",
    filename="parameters",
    extension="csv",
    compression=None,
    level=0,
):
    """parameters estimation function"""

    # reindexing
    moments = moments.reset_index().set_index(["code", "duration", "block"])

    # dataframe
    dataframe = pandas.DataFrame(index=moments.index)

    # entry update
    dataframe.loc[:, ["τ", "κ"]] = moments.loc[:, ["τ", "κ"]]
    dataframe.loc[:, ["n"]] = (
        moments.loc[:, ["λ"]].to_numpy() / moments.loc[:, ["δ"]].to_numpy()
    )

    if regional:
        dataframe.loc[:, ["T", "K"]] = moments.loc[:, ["T", "K"]]
        dataframe.loc[:, ["N"]] = (
            moments.loc[:, ["Λ"]].to_numpy() / moments.loc[:, ["Δ"]].to_numpy()
        )

    # parameters estimation
    dataframe.loc[:, labels] = dists.parameters(
        distribution,
        *arguments,
        array=moments.loc[:, orders].to_numpy(),
        **keywords,
    )

    if "include" in dataframe.columns:
        dataframe.loc[:, ["include"]] = moments.loc[:, ["include"]]

    if filename is not None:
        utils.save(dataframe, filename, outdir, extension, compression)

    return dataframe


def projector(
    dataframe,
    station,
    duration,
    size="λ",
    labels=["L1", "L2", "L3"],
    clusters=None,
    infer=None,
    blocks=None,
    regional=None,
    coherency=True,
    level=0,
):
    """projector function"""

    # reindexing
    dataframe = dataframe.reset_index().set_index(
        ["code", "duration", "block"]
    )

    if clusters is None:
        vector = (
            dataframe.loc[(station, duration), size]
            .to_numpy()
            .reshape((-1, 1, 1))
        )
        out = (
            dataframe.loc[(station, duration), labels]
            .to_numpy()
            .reshape((-1, len(labels), 1))
        )

    else:
        if type(clusters).__name__ == "Index":
            array = dataframe.index.get_level_values("code").intersection(
                clusters
            )
        elif type(clusters).__name__ == "DataFrame":
            array = dataframe.index.get_level_values("code").intersection(
                pandas.Index(clusters.loc[station, "array"])
            )

        if infer is None:
            array = array.union([station])

        if blocks is None:
            vector = (
                dataframe.loc[(array, duration), size]
                .to_numpy()
                .reshape((1, 1, -1))
            )
            out = (
                dataframe.loc[(array, duration), labels]
                .to_numpy()
                .reshape((1, len(labels), -1))
            )

        else:
            # blocks aggregate
            aggregate = (
                dataframe.loc[(array, duration), :]
                .index.get_level_values("block")
                .drop_duplicates()
                .sort_values()
            )

            # initializing arrays
            vector = numpy.full((len(aggregate), 1, len(array)), numpy.nan)
            out = numpy.full(
                (len(aggregate), len(labels), len(array)), numpy.nan
            )

            for number, station in enumerate(array):
                # block values
                block = dataframe.loc[
                    (station, duration), :
                ].index.get_level_values("block")

                # blocks indexes
                index = aggregate.searchsorted(block)

                # assign
                vector[index, :, number] = (
                    dataframe.loc[(station, duration, block), size]
                    .to_numpy()
                    .reshape((-1, 1))
                )
                out[index, :, number] = (
                    dataframe.loc[(station, duration, block), labels]
                    .to_numpy()
                    .reshape((-1, len(labels)))
                )

    if coherency:
        vector[numpy.expand_dims(numpy.isnan(out[:, 0, :]), 1)] = numpy.nan

    if regional:
        out = numpy.nansum(out * vector, axis=2) / numpy.nansum(vector, axis=2)

    return vector, out


def empiricals(
    dtb,
    values,
    stations=None,
    durations=["01m", "01h", "01d"],
    outdir="data",
    filename="empiricals",
    extension="csv",
    compression=None,
    cpu=1,
    **keywords,
):
    """empiricals evaluation function"""

    if stations is None:
        if type(dtb).__name__ in ["Series", "DataFrame"]:
            stations = dtb.index
        elif isinstance(dtb, str):
            stations = file.listing(dtb, output="filename")
        elif isinstance(dtb, (list, tuple)):
            stations = file.listing(*dtb, output="filename")
        elif isinstance(dtb, (dict)):
            stations = file.listing(**dtb, output="filename")
        else:
            raise

    # indexers
    stations, durations = utils.indexer(stations, durations)

    if isinstance(values, (str, list, tuple)):
        stations = stations.intersection(file.listing(values, "filename"))
    else:
        stations = stations.intersection(values.index)

    # defining function
    function = functools.partial(sites.empiricals, values, **keywords)

    # processing results
    results = utils.process(function, stations, durations, cpu=cpu)

    if results is None:
        return

    # dataframe
    dataframe = pandas.concat(results).set_index(["code", "duration", "block"])

    if filename is not None:
        utils.save(dataframe, filename, outdir, extension, compression)

    return dataframe


def quantiles(
    dtb,
    values,
    parameters,
    stations=None,
    durations=["01m", "01h", "01d"],
    outdir="data",
    filename="quantiles",
    extension="csv",
    compression=None,
    cpu=1,
    **keywords,
):
    """quantiles evaluation function"""

    if stations is None:
        if type(dtb).__name__ in ["Series", "DataFrame"]:
            stations = dtb.index
        elif isinstance(dtb, str):
            stations = file.listing(dtb, output="filename")
        elif isinstance(dtb, (list, tuple)):
            stations = file.listing(*dtb, output="filename")
        elif isinstance(dtb, (dict)):
            stations = file.listing(**dtb, output="filename")
        else:
            raise

    # reindexing
    parameters = parameters.reset_index().set_index(
        ["code", "duration", "block"]
    )

    # indexers
    stations, durations = utils.indexer(stations, durations)

    # defining function
    function = functools.partial(
        sites.quantiles, values, parameters, **keywords
    )

    # processing results
    results = utils.process(function, stations, durations, cpu=cpu)

    if results is None:
        return

    # dataframe
    dataframe = pandas.concat(results)

    if "block" in dataframe.columns:
        dataframe.set_index(["code", "duration", "block"], inplace=True)
    else:
        dataframe.set_index(["code", "duration"], inplace=True)

    if filename is not None:
        utils.save(dataframe, filename, outdir, extension, compression)

    return dataframe


def jocker(
    dtb,
    values,
    stations=None,
    durations=["01m", "01h", "01d"],
    benchmarks="site",
    outdir="data",
    filename="metrics",
    extension="csv",
    compression=None,
    subdir=None,
    subdic={},
    procdir="validation",
    procdic={},
    procset=None,
    cpu=1,
    **keywords,
):
    """cross-validator function"""

    if stations is None:
        if type(dtb).__name__ in ["Series", "DataFrame"]:
            stations = dtb.index
        elif isinstance(dtb, str):
            stations = file.listing(dtb, output="filename")
        elif isinstance(dtb, (list, tuple)):
            stations = file.listing(*dtb, output="filename")
        elif isinstance(dtb, (dict)):
            stations = file.listing(**dtb, output="filename")
        else:
            raise

    # indexers
    stations, durations, benchmarks = utils.indexer(
        stations, durations, benchmarks
    )

    if isinstance(values, (str, list, tuple)) and subdir is None:
        subdir = values

    if procset == "difference":
        stations = utils.indexer(
            [
                i
                for i in stations
                if not any(i in j for j in file.listing(procdir, "filename"))
            ]
        )
    elif procset == "intersection":
        stations = utils.indexer(
            [
                i
                for i in stations
                if any(i in j for j in file.listing(procdir, "filename"))
            ]
        )

    # defining function
    function = functools.partial(
        sites.jocker,
        values,
        subdir=subdir,
        subdic=subdic,
        procdir=procdir,
        procdic=procdic,
        **keywords,
    )

    # processing results
    results = utils.process(function, stations, durations, benchmarks, cpu=cpu)

    if results is None:
        return

    # dataframe
    dataframe = pandas.concat(results)

    if filename is not None:
        utils.save(dataframe, filename, outdir, extension, compression)

    return dataframe
