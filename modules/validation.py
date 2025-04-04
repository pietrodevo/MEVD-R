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

import numpy
import pandas
import random

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import dists
import stats
import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def matrix(array, dataframe, column="block"):
    """matrix function"""

    # blocks aggregate
    aggregate = dataframe.loc[array, column].drop_duplicates().sort_values()

    # initializing matrix
    matrix = numpy.full((len(aggregate), 1, len(array)), numpy.nan)

    for number, member in enumerate(array):
        # block values
        values = dataframe.loc[member, column]

        # block indexes
        index = aggregate.searchsorted(values)

        # assign
        matrix[index, :, number] = values.to_numpy().reshape(-1, 1)

    return matrix


def shuffler(
    matrix,
    skip=None,
    size=None,
    samples=1e3,
    unique="arrays",
    full=None,
    refill=None,
    repeat=None,
    reshape=None,
):
    """shuffler function"""

    # dimensions
    length, vectors = matrix.shape[0], matrix.shape[1]

    if size is None:
        size = length

    if not isinstance(samples, int):
        samples = int(samples)

    if type(skip).__name__ == "ndarray":
        skip = skip.reshape(((-1, samples)))

    if full:
        if refill:
            # filter for all values indexes in the matrix
            m_filter = list(numpy.arange(length))
        else:
            # filter for non-nan values indexes in the matrix
            m_filter = list(
                numpy.where(numpy.invert(numpy.isnan(matrix).any(axis=1)))[0]
            )

    if full:
        # filter from matrix
        v_filter = [m_filter.copy() for vector in range(vectors)]
    elif refill:
        # filter for all values indexes in the vector
        v_filter = [list(numpy.arange(length)) for vector in range(vectors)]
    else:
        # filter for non-nan values indexes in the vector
        v_filter = [
            list(numpy.where(~numpy.isnan(matrix[:, vector]))[0])
            for vector in range(vectors)
        ]

    if unique == "vectors":

        # initialize indexer
        indexer = numpy.full((length, vectors, samples), numpy.nan)

        for vector in range(vectors):

            # resetting seed
            random.seed()

            if repeat is None:
                if skip is None:
                    v_samples = [
                        random.sample(v_filter[vector], k=size)
                        for i in range(samples)
                    ]
                else:
                    v_samples = [
                        random.sample(
                            [
                                v
                                for v in v_filter[vector]
                                if v not in numpy.where(skip[:, i] == 1)[0]
                            ],
                            k=size,
                        )
                        for i in range(samples)
                    ]
            else:
                if skip is None:
                    v_samples = [
                        random.choices(v_filter[vector], k=size)
                        for i in range(samples)
                    ]
                else:
                    v_samples = [
                        random.choices(
                            [
                                [
                                    v
                                    for v in v_filter[vector]
                                    if v not in numpy.where(skip[:, i] == 1)[0]
                                ]
                            ],
                            k=size,
                        )
                        for i in range(samples)
                    ]

            for sample in range(samples):
                # vector indexers/counters
                v_index, v_counts = numpy.unique(
                    v_samples[sample], return_counts=True
                )
                # assigning indexers/counters
                indexer[v_index, vector, sample] = v_counts

    if unique == "arrays":

        # initialize
        arrays = []

        for sample in range(samples):

            while True:

                # resetting seed
                random.seed()

                # initialize extraction array
                array = numpy.full((length, vectors), numpy.nan)

                for vector in range(vectors):

                    if repeat is None:
                        if skip is None:
                            v_sample = random.sample(v_filter[vector], k=size)
                        else:
                            v_sample = random.sample(
                                [
                                    v
                                    for v in v_filter[vector]
                                    if v
                                    not in numpy.where(skip[:, sample] == 1)[0]
                                ],
                                k=size,
                            )
                    else:
                        if skip is None:
                            v_sample = random.choices(v_filter[vector], k=size)
                        else:
                            v_sample = random.choices(
                                [
                                    v
                                    for v in v_filter[vector]
                                    if v
                                    not in numpy.where(skip[:, sample] == 1)[0]
                                ],
                                k=size,
                            )

                    # vector indexers/counters
                    v_index, v_counts = numpy.unique(
                        v_sample, return_counts=True
                    )

                    # assigning indexers/counters
                    array[v_index, vector] = v_counts

                if not any(numpy.array_equal(array, a) for a in arrays):
                    break

            # appending matrix
            arrays.append(array)

        # generating indexer
        indexer = numpy.dstack(arrays)

    if reshape and vectors == 1:
        indexer = indexer.reshape((-1, samples))

    return indexer


def empiricals(
    data,
    indexer,
    blocks,
    normalize=None,
    invert=None,
    maxima=None,
    function=dists.empirical,
    **keywords,
):
    """empiricals function"""

    # converting nans
    indexer[numpy.isnan(indexer)] = 0

    # setting type
    indexer = indexer.astype(int)

    # initializing empiricals
    empiricals = []

    if normalize is not None:
        data = data / data.mean()

    if not callable(function):
        if isinstance(function, str):
            function = utils.definition(function)
        elif isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)

    for extraction in range(indexer.shape[1]):
        # extraction block
        block = numpy.repeat(blocks[:, :].reshape(-1), indexer[:, extraction])

        if invert is None:
            subset = data.loc[data.index.year.isin(block)]
        else:
            subset = data.loc[numpy.invert(data.index.year.isin(block))]

        if maxima:
            subset = subset.resample(maxima).max().dropna().to_numpy()

        # empirical dataframe
        empiricals.append(function(subset, **keywords))

    return empiricals


def extractor(
    n,
    array,
    indexer,
    regional=None,
    distribution=None,
    coefficients=None,
    function=dists.parameters,
    **keywords,
):
    """extractor function"""

    # number of extractions
    extractions = indexer.shape[2]

    # generating indexers
    indexer_n = indexer.reshape(
        (n.shape[0], n.shape[1], n.shape[2], extractions)
    )
    indexer_array = numpy.hstack([indexer_n] * array.shape[1])

    if numpy.any(indexer > 1):
        # generating lists
        list_n = [
            numpy.dstack(
                [
                    numpy.hstack(
                        [
                            n[:, k, j]
                            .repeat(
                                numpy.nan_to_num(indexer_n[:, k, j, i]).astype(
                                    int
                                )
                            )
                            .reshape(-1, 1)
                            for k in range(indexer_n.shape[1])
                        ]
                    ).reshape(-1, indexer_n.shape[1], 1)
                    for j in range(indexer_n.shape[2])
                ]
            )
            for i in range(extractions)
        ]
        list_array = [
            numpy.dstack(
                [
                    numpy.hstack(
                        [
                            array[:, k, j]
                            .repeat(
                                numpy.nan_to_num(
                                    indexer_array[:, k, j, i]
                                ).astype(int)
                            )
                            .reshape(-1, 1)
                            for k in range(indexer_array.shape[1])
                        ]
                    ).reshape(-1, indexer_array.shape[1], 1)
                    for j in range(indexer_array.shape[2])
                ]
            )
            for i in range(extractions)
        ]

    else:
        # generating lists
        list_n = [n * indexer_n[:, :, :, i] for i in range(extractions)]
        list_array = [
            array * indexer_array[:, :, :, i] for i in range(extractions)
        ]

    if coefficients is None:
        c = 1
    else:
        c = stats.normalize(coefficients, reshape=True)

    if regional:
        # generating lists
        list_array = [
            numpy.nansum(
                c * n * array,
                axis=2,
            )
            / numpy.nansum(c * n, axis=2)
            for n, a in zip(list_n, list_array)
        ]
        list_n = [
            numpy.nanmean(c * n, axis=2) / numpy.nanmean(c) for n in list_n
        ]

    if not callable(function):
        if isinstance(function, str):
            function = utils.definition(function)
        elif isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)

    if distribution is not None:
        if regional is None:
            list_array = [
                numpy.dstack(
                    [
                        function(distribution, array=array[..., i], **keywords)
                        for i in range(array.shape[-1])
                    ]
                )
                for array in list_array
            ]

        else:
            list_array = [
                function(distribution, array=array, **keywords)
                for array in list_array
            ]

    if regional:
        axis = 2
    else:
        axis = 3

    return numpy.stack(list_n, axis), numpy.stack(list_array, axis)


def estimator(
    data,
    indexer,
    blocks,
    column=None,
    maxima=None,
    threshold=None,
    normalize=None,
    delta=None,
    distribution=None,
    coefficients=None,
    function=dists.parameters,
    outputs=["L1", "L2", "L3"],
    **keywords,
):
    """estimator function"""

    # index
    index = data.index.unique()

    # number of extractions/members
    extractions, members = indexer.shape[2], blocks.shape[2]

    # indexing
    threshold, normalize = utils.indexer(threshold, normalize)

    # intializing moments
    moments = numpy.ones((extractions, len(outputs))) * numpy.nan

    if not callable(function):
        if isinstance(function, str):
            function = utils.definition(function)
        elif isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)

    # initializing lists
    n, par = [], []

    for extraction in range(extractions):
        # initializing array arrays
        array_n = numpy.full((members, 1), numpy.nan)
        array_moments = numpy.full((members, len(outputs)), numpy.nan)

        for member in range(members):
            # station block
            m_block = blocks[:, :, member].reshape(-1).astype(int)

            # blocks occurencies
            m_occurencies = numpy.nan_to_num(
                indexer[:, member, extraction]
            ).astype(int)

            # member data
            m_data = (
                data.loc[index[member], :]
                .set_index(
                    pandas.DatetimeIndex(data.loc[index[member], "datetime"])
                )
                .drop(columns="datetime")
            )

            # data indexer
            m_index = m_block.repeat(m_occurencies).astype(str)

            # data sample
            m_sample = pandas.concat([m_data.loc[index] for index in m_index])

            # member stats
            out = stats.blocks(
                m_sample,
                column=column,
                maxima=maxima,
                threshold=threshold[member],
                normalize=normalize[member],
                delta=delta,
                outputs=outputs,
                statistics=True,
                dataframe=False,
            )

            # assignments
            array_n[member, :] = out[5] / out[1]
            array_moments[member, :] = out[-1]

        # extraction moments
        moments[extraction, :] = stats.swm(
            scale=None,
            size=array_n,
            orders=array_moments,
            coefficients=coefficients,
            dataframe=False,
        )

        if distribution is not None:
            n.append(array_n.mean())
            par.append(
                function(
                    distribution,
                    array=moments[extraction, :].reshape(1, -1),
                    **keywords,
                )
            )

    if distribution is not None:
        return numpy.vstack(n), numpy.vstack(par)
    else:
        return moments


def evaluator(
    empiricals,
    distribution,
    *args,
    axis=0,
    check=None,
    function=dists.quantile,
    label="value",
    **keywords,
):
    """evaluator function"""

    # extractions
    extractions = min([arg.shape[axis] for i, arg in enumerate(args)])

    if not isinstance(empiricals, list):
        empiricals = [empiricals] * extractions

    if axis == 0:
        iterables = [
            tuple([arg[extraction, ...] for arg in args])
            for extraction in range(extractions)
        ]
    else:
        iterables = [
            tuple([arg[..., extraction] for arg in args])
            for extraction in range(extractions)
        ]

    if not callable(function):
        if isinstance(function, str):
            function = utils.definition(function)
        elif isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)

    # initialize
    evaluations, flags = [], []

    for extraction, empirical, arg in zip(
        range(1, extractions + 1), empiricals, iterables
    ):
        # computing quantiles
        quantile = function(empirical, distribution, *arg, **keywords)

        if check and not quantile.loc[:, label].isnull().values.any():
            flags.append(extraction)

        # appending quantiles
        evaluations.append(
            dists.evaluation(empirical, quantile).set_index(
                numpy.ones(quantile.shape[0], dtype=int) * extraction
            )
        )

    # concatenating
    dataframe = pandas.concat(evaluations).rename_axis("extraction")

    if check is None:
        return dataframe
    else:
        return dataframe, flags
