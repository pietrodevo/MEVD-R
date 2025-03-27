# -*- coding: utf-8 -*-

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MANIFEST"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

"""

author: pietro devò
e-mail: pietro.devo@dicea.unipandas.it

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

import numpy
import pandas

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import dates
import file
import readers
import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def series(
    filename,
    subdir="series",
    extension="csv",
    compression="infer",
    subdic={},
    split=None,
    reader=None,
    skipspaces=True,
    skipheader=0,
    skipfooter=0,
    variables=None,
    separator=None,
    dateparser=True,
    dateformat="%Y-%m-%d %H:%M:%S",
    thousands=None,
    decimal=".",
    flags=None,
    rename=None,
    index=None,
    nans=None,
    dtype=None,
    keep=None,
    scale=None,
    continuous=None,
    resample=None,
    interpolate=None,
    sort=None,
    drop=None,
    **keywords,
):
    """series loading

    filename    -> input filename;
    subdir      -> input subdir;
    extension   -> input extension;
    compression -> input compression;
    subdic      -> input dictionary;
    split       -> optional splitting at directory/filename level;
    reader      -> reader function or string/list/tuple/dict for definition;
    skipspaces  -> skip spaces after delimiter;
    skipheader  -> number of lines at the start of file to skip;
    skipfooter  -> number of lines at the end of file to skip;
    variables   -> selected variable(s);
    separator   -> column separation;
    dateparser  -> date parsing;
    dateformat  -> date formatting;
    thousands   -> character for thousands;
    decimal     -> character for decimals;
    flags       -> validty flags dictionary;
    rename      -> renaming dictionary;
    index       -> column for index;
    nans        -> replacing nans values;
    dtype       -> selected type(s);
    keep        -> keeping label(s);
    scale       -> scaling coefficient(s);
    continuous  -> missing nans;
    resample    -> resampling frequency;
    interpolate -> missing data interpolation;
    sort        -> sorting flag;
    drop        -> dropping function;
    keywords    -> reader additional keywords.

    """

    if split is None:
        path = [file.path(filename, subdir, extension, **subdic)]
    else:
        path = file.split(filename, subdir, extension, split, **subdic)
        if len(path) == 0:
            return

    if variables is not None:
        if isinstance(variables, (int, float, str)):
            variables = [variables]
        elif not isinstance(variables, list):
            variables = list(variables)

    if dateparser is not None:
        if dateparser is True:
            if index is not None:
                if isinstance(rename, dict) and index in rename.values():
                    dateparser = list(rename.keys())[
                        list(rename.values()).index(index)
                    ]
                else:
                    dateparser = index
            else:
                dateparser = None
        if isinstance(dateparser, (int, float, str)):
            dateparser = [dateparser]
        elif not isinstance(dateparser, (type(None), list, tuple)):
            dateparser = list(dateparser)

    if reader is None:

        if extension == "pkl":

            # reading data
            data = [
                file.load(
                    i,
                    subdir=None,
                    extension=None,
                    compression=compression,
                )
                for i in path
            ]

        else:

            if variables is not None:
                usecols = variables.copy()
                if dateparser is not None:
                    usecols = usecols + dateparser
                if isinstance(flags, dict):
                    usecols = usecols + list(flags.keys())
                if index is not None:
                    if not isinstance(rename, dict) and index not in usecols:
                        usecols.append(index)
                    elif (
                        isinstance(rename, dict)
                        and index in rename.values()
                        and list(rename.keys())[
                            list(rename.values()).index(index)
                        ]
                        not in usecols
                    ):
                        usecols.append(
                            list(rename.keys())[
                                list(rename.values()).index(index)
                            ]
                        )

            else:
                usecols = None

            # reading data
            data = [
                pandas.read_csv(
                    i,
                    compression=compression,
                    skipinitialspace=skipspaces,
                    skiprows=skipheader,
                    skipfooter=skipfooter,
                    sep=separator,
                    usecols=usecols,
                    thousands=thousands,
                    decimal=decimal,
                    **keywords,
                )
                for i in path
            ]

    else:
        if isinstance(reader, str):
            reader = utils.definition(readers, reader)
        elif isinstance(reader, (list, tuple)):
            reader = utils.definition(*reader)
        elif isinstance(reader, dict):
            reader = utils.definition(**reader)

        # reading data
        data = [reader(i, variables, **keywords) for i in path]

    # concatenate data
    data = pandas.concat(data)

    if isinstance(flags, dict):
        for key, value in flags.items():
            if not isinstance(value, (list, tuple, range)):
                value = [value]
            data = data.loc[
                data.loc[:, key].isin(
                    [i if i != None else numpy.nan for i in value]
                )
            ]

    if dateparser is not None:
        # parsing dates
        data.loc[:, index] = (
            data.loc[:, dateparser].astype(str).agg(" ".join, axis=1)
        )

        # removing columns
        data.drop(columns=dateparser, inplace=True)

    if isinstance(rename, dict):
        # renaming
        data.rename_axis(index=rename, inplace=True)
        data.rename(columns=rename, inplace=True)

        # variables
        variables = [rename[i] if i in rename.keys() else i for i in variables]

        if isinstance(interpolate, (int, str)):
            interpolate = [interpolate]
        if isinstance(interpolate, (list, tuple)):
            interpolate = [
                rename[i] if i in rename.keys() else i for i in interpolate
            ]

        if isinstance(dtype, dict):
            dtype = {
                rename[i] if i in rename else i: j for i, j in dtype.items()
            }

    if index is not None and data.index.name != index:
        data.set_index(index, inplace=True)

        if isinstance(data.index, object) and dateformat is not None:
            data.index = pandas.to_datetime(data.index, format=dateformat)

    if nans:
        data.replace(nans, numpy.nan, inplace=True)

    if dtype:
        data = data.astype(dtype)

    if keep is not None:
        data = data.loc[:, keep]
    elif keep is None and variables is not None:
        data = data.loc[:, variables]

    if isinstance(scale, (int, float)):
        data = data * scale
    elif isinstance(scale, dict) and any(
        [i in (data.columns) for i in scale.keys()]
    ):
        data.loc[:, scale.keys()] = data.loc[:, scale.keys()].mul(
            scale.values()
        )

    if continuous:
        data = data.asfreq(dates.frequency(data))

    if resample is not None:
        data = data.resample(resample).sum(numeric_only=True)

    if interpolate:
        data.loc[:, slice(None) if interpolate is True else interpolate] = (
            data.loc[
                :, slice(None) if interpolate is True else interpolate
            ].interpolate()
        )

    if sort:
        data.sort_index(inplace=True)

    if drop:
        data.dropna(how=drop, inplace=True)

    return data


def filtering(
    dataframe,
    frequency=None,
    column="value",
    blocks="y",
    percentile=0.9,
    length=5,
):
    """filtering data

    dataframe  -> input dataframe;
    frequency  -> time frequency;
    column     -> data column;
    blocks     -> time blocks;
    percentile -> filtering availability;
    length     -> filtering length.

    """

    if frequency is None:
        frequency = dates.frequency(dataframe)

    # adjustments
    frequency = pandas.Timedelta(frequency)

    # blocks counting
    count = dataframe.loc[:, column].resample(blocks).count()

    # lowering strings
    blocks = blocks.lower()

    if blocks == "y":
        index = dataframe.index.year
        count.index = count.index.year
        delta = 365
    if blocks == "m":
        index = dataframe.index.month
        count.index = count.index.month
        delta = 30
    if blocks == "d":
        index = dataframe.index.day
        count.index = count.index.day
        delta = 1

    # index coherency
    count = count.loc[count.index.intersection(index)]

    if percentile is not None:
        dataframe = dataframe.loc[
            numpy.isin(
                index,
                index.unique()[
                    count
                    >= int(pandas.Timedelta(delta, "d") / frequency)
                    * percentile
                ],
            )
        ]
        if dataframe.shape[0] == 0:
            raise

    if length is not None:
        if blocks == "y":
            index = dataframe.index.year
        if blocks == "m":
            index = dataframe.index.month
        if blocks == "d":
            index = dataframe.index.day

        if index.unique().size < length:
            raise

    return dataframe


def storms(
    dataframe,
    frequency=None,
    column="value",
    lag=None,
    threshold=None,
    durations=None,
    output=None,
):
    """storms separation analyses

    dataframe -> input dataframe;
    frequency -> time frequency;
    column    -> data column;
    lag       -> time lag;
    threshold -> minimum value;
    durations -> computed duration(s);
    output    -> optional output filter.

    """

    if durations is None:
        durations = list(dataframe.columns)
    if not isinstance(durations, list):
        durations = [durations]

    if frequency is None:
        frequency = dates.frequency(dataframe)

    # adjustments
    frequency = pandas.Timedelta(frequency)

    if lag is None:
        lag = frequency
    else:
        lag = pandas.Timedelta(lag)

    # indexing
    steps = numpy.asarray([d / frequency for d in durations], dtype=int)
    delta = int(lag / frequency)

    # data vector
    data = dataframe.loc[:, column].to_numpy()

    # rolling subranges
    rolling = numpy.lib.stride_tricks.as_strided(
        data, shape=(data.size - delta + 1, delta), strides=data.strides * 2
    )

    if threshold is None:
        threshold = 0

    # pointers of [delta]*threshold patterns
    pointers = numpy.concatenate(
        (
            [-delta],
            numpy.where(numpy.all(rolling < [threshold] * delta, axis=1))[0],
            [data.size],
        )
    )

    # sequences of [delta]*threshold patterns
    sequences = numpy.stack((pointers[:-1] + delta, pointers[1:] - 1)).T

    # indexing
    index = sequences[sequences[:, 0] <= sequences[:, 1]]

    # initialize dictionary
    dictionary = {
        i: {"flag": None, "array": {}, "maxima": {}, "datetime": {}}
        for i in range(len(index))
    }

    # initialize dataframe
    maxima = pandas.DataFrame(columns=durations, dtype=float)
    maxima.index.name = dataframe.index.name

    for i, (start, end) in enumerate(index):
        dictionary[i]["start"] = dataframe.index[start]
        dictionary[i]["end"] = dataframe.index[end]
        dictionary[i]["span"] = (
            dictionary[i]["end"] - dictionary[i]["start"] + frequency
        )
        dictionary[i]["data"] = dataframe.loc[
            dictionary[i]["start"] : dictionary[i]["end"], column
        ].to_numpy()
        dictionary[i]["nan"] = numpy.isnan(dictionary[i]["data"]).any()
        dictionary[i]["inf"] = numpy.isinf(dictionary[i]["data"]).any()
        dictionary[i]["size"] = dictionary[i]["data"].size

        if (
            dictionary[i]["size"] > 0
            and not dictionary[i]["nan"]
            and not dictionary[i]["inf"]
        ):
            dataframe.loc[
                dictionary[i]["start"] : dictionary[i]["end"], "event"
            ] = True

            for s, d in zip(steps, durations):
                if dictionary[i]["size"] >= s:
                    dictionary[i]["array"][d] = numpy.nansum(
                        numpy.lib.stride_tricks.as_strided(
                            dictionary[i]["data"],
                            shape=(dictionary[i]["size"] - s + 1, s),
                            strides=dictionary[i]["data"].strides * 2,
                        ),
                        axis=1,
                    )
                    dictionary[i]["maxima"][d] = dictionary[i]["array"][
                        d
                    ].max()
                    dictionary[i]["datetime"][d] = (
                        str(
                            dataframe.index[
                                start + dictionary[i]["array"][d].argmax()
                            ]
                        )
                        + " -> "
                        + str(
                            dataframe.index[
                                dictionary[i]["array"][d].argmax() + s
                            ]
                        )
                    )
                else:
                    dictionary[i]["maxima"][d] = numpy.nan

                if dictionary[i]["maxima"][d] == 0:
                    dictionary[i]["maxima"][d] = numpy.nan

            if numpy.nan not in dictionary[i]["maxima"].values():
                dictionary[i]["flag"] = True

            else:
                dictionary[i]["flag"] = False

            maxima.loc[dictionary[i]["start"], :] = list(
                dictionary[i]["maxima"].values()
            )

        else:
            dataframe.loc[
                dictionary[i]["start"] : dictionary[i]["end"], "event"
            ] = False

    if output is None:
        return dataframe, maxima, dictionary
    elif output is True:
        return {
            "dataframe": dataframe,
            "maxima": maxima,
            "dictionary": dictionary,
        }
    elif isinstance(output, str):
        return locals()[output]
