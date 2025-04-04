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
import scipy

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import dates
import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def normalize(x, axis=0, collapse=1, function=numpy.nanmean, reshape=None):
    """input normalize function"""

    if type(x).__name__ in ["Series", "DataFrame"]:
        x = x.to_numpy()
    elif type(x).__name__ != "ndarray":
        x = numpy.stack(
            [numpy.array(i).squeeze() for i in x],
            axis=1,
        )
    elif x.ndim == 1:
        x = x.reshape(-1, 1)

    # normalizing values
    x = x / numpy.nansum(x, axis=axis, keepdims=True)

    if not callable(function):
        if isinstance(function, str):
            function = utils.definition(function)
        elif isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)
        else:
            function = None

    if callable(function):
        x = function(x, axis=collapse, keepdims=True) / numpy.nansum(
            function(x, axis=collapse, keepdims=True), axis=axis, keepdims=True
        )

    if reshape:
        return x.squeeze()
    else:
        return x


def quantile(data, value, drop=True):
    """data quantile function"""

    if drop:
        data = data[~numpy.isnan(data)]

    if isinstance(value, float) and (0 <= value <= 1):
        return numpy.quantile(data, value)
    elif isinstance(value, str) and ("." in value):
        return numpy.quantile(data, float(value))
    elif isinstance(value, str) and ("%" in value):
        return numpy.percentile(
            data, float(int("".join(i for i in value if i.isdigit())))
        )


def trend(data, function="linear", **keywords):
    """data trend function"""

    if type(function).__name__ != "function":
        if function is None:
            raise
        elif function == "linear":
            function = lambda x, a, b: a * x + b
        elif function == "quadratic":
            function = lambda x, a, b, c: a * x**2 + b * x + c

    # generating x vector
    x = numpy.arange(data.shape[0])

    # fitting function to the data
    par, opt = scipy.optimize.curve_fit(
        function, x[~numpy.isnan(data)], data[~numpy.isnan(data)], **keywords
    )

    return function(x, *par)


def pwm(data):
    """sample probability weighted moments"""

    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # initialize
    pw = numpy.full((data.shape[1], 3), numpy.nan)

    for i, d in enumerate(data.T):
        # sorting data
        X = numpy.sort(d)

        # data size
        s = X.size

        # compute pw-moments
        pw[i, :] = X.mean(), 0, 0

        for j in range(1, s):
            pw[i, 1] = pw[i, 1] + (j) / (s - 1) * X[j]
            pw[i, 2] = pw[i, 2] + (j) * (j - 1) / (s - 1) / (s - 2) * X[j]

        # normalization
        pw[i, [1, 2]] = pw[i, [1, 2]] / s

    return pw


def lm(data, method="pwm", order=3, array=True):
    """sample linear moments"""

    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # initialize
    L = numpy.full((data.shape[1], order), numpy.nan)

    for i, d in enumerate(data.T):

        if method == "pwm" and order <= 3:
            # compute pw-moments
            pw = pwm(d)

            # compute l-moments as linear combination of the pwms
            if order >= 1:
                L[i, 0] = pw[:, 0]
            if order >= 2:
                L[i, 1] = 2 * pw[:, 1] - pw[:, 0]
            if order >= 3:
                L[i, 2] = 6 * pw[:, 2] - 6 * pw[:, 1] + pw[:, 0]

        if method == "legendre" or order > 3:

            # sorting data
            X = numpy.sort(d)

            # data size
            n = X.size

            # initialize
            b = numpy.zeros(order - 1)

            # first order moment
            L[i, 0] = X.mean()

            for r in range(1, order):
                N = numpy.prod(
                    numpy.tile(numpy.arange(r + 1, n + 1), (r, 1))
                    - numpy.tile(
                        numpy.arange(1, r + 1).reshape(-1, 1), (1, n - r)
                    ),
                    axis=0,
                )
                D = numpy.prod(numpy.tile(n, r) - numpy.arange(1, r + 1))
                b[r - 1] = 1 / n * numpy.add.reduce(N / D * X[r:n], axis=0)

            # stacking statistics
            B = numpy.hstack((L[i, 0], b))[::-1]

            for j in range(1, order):
                z = numpy.zeros(len(B) - (j + 1))
                s = scipy.special.sh_legendre(j)
                C = numpy.hstack((z, s))
                L[i, j] = numpy.add.reduce(C * B, axis=0)

    if array:
        return L
    else:
        if L.size == 1:
            return L[0]
        else:
            return tuple(i.reshape(-1, 1) for i in L.T)


def swm(
    data=None,
    index=None,
    orders=3,
    scale="κ",
    size="λ",
    coefficients=None,
    reshape=True,
    dataframe=True,
):
    """sample scaled and weighted moments"""

    if scale is None:
        κ = 1
    elif (
        isinstance(scale, (int, float, list, tuple))
        or type(scale).__name__ == "ndarray"
    ):
        κ = scale
    if size is None:
        λ = 1
    elif (
        isinstance(size, (int, float, list, tuple))
        or type(size).__name__ == "ndarray"
    ):
        λ = size

    if type(data).__name__ == "DataFrame":
        if index is None:
            index = slice(None)

        if isinstance(orders, int):
            orders = ["L" + str(orders)]
        elif isinstance(orders, str):
            orders = utils.enumerating(orders)

        # orders
        m = data.loc[index, orders].to_numpy()

        if isinstance(scale, (int, str)):
            κ = data.loc[index, scale].to_numpy().reshape(-1, 1)
        if isinstance(size, (int, str)):
            λ = data.loc[index, size].to_numpy().reshape(-1, 1)

        if isinstance(coefficients, (int, str, list, tuple)):
            coefficients = data.loc[index, coefficients].to_numpy()

    elif type(orders).__name__ == "ndarray":
        # linear moments
        m = orders

        # linear orders
        orders = utils.enumerating("L" + str(orders.shape[1]))

    if reshape and m.ndim != 1:
        if type(κ).__name__ == "ndarray":
            κ = κ.reshape(-1, 1)
        if type(λ).__name__ == "ndarray":
            λ = λ.reshape(-1, 1)

    if coefficients is None:
        c = 1
    else:
        c = normalize(coefficients)

    # computing stats
    x = (
        numpy.add.reduce(c * m * λ, axis=0)
        / numpy.add.reduce(c * κ * λ, axis=0)
    ).reshape(-1, m.shape[1])

    if dataframe:
        return pandas.DataFrame(columns=orders, data=c)
    elif x.shape[0] == 1:
        return x[0]
    else:
        return x


def blocks(
    data,
    index=None,
    column=None,
    blocks=None,
    maxima=None,
    threshold=None,
    normalize=None,
    lenght=3,
    delta=None,
    function=lm,
    outputs=3,
    arguments=(),
    keywords={},
    statistics=True,
    dataframe=True,
):
    """blocks function"""

    if not callable(function):
        if isinstance(function, str):
            function = utils.definition(function)
        elif isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)
        else:
            function = None

    if function is None:
        outputs = []
    elif isinstance(outputs, int):
        outputs = [
            ("L" if function.__name__ == "lm" else "X") + str(i)
            for i in range(1, outputs + 1)
        ]
    elif isinstance(outputs, str):
        outputs = utils.enumerating(outputs)

    if type(data).__name__ in ["Series", "DataFrame"]:

        if index is not None:
            data = data.set_index(
                pandas.DatetimeIndex(data.loc[:, index])
            ).drop(columns=index)
        else:
            data.index = pandas.DatetimeIndex(data.index)

        if column is not None:
            data = data.loc[:, column]
        else:
            data = data.squeeze()

        if blocks is None:
            unit = "y"
            step = numpy.subtract.reduce(data.index.year.unique()[[-1, 0]]) + 1
        else:
            unit = (
                str("".join(i for i in blocks if not i.isdigit()))
                if any(not i.isdigit() for i in blocks)
                else "y"
            )
            step = (
                int("".join(i for i in blocks if i.isdigit()))
                if any(i.isdigit() for i in blocks)
                else 1
            )

        # array of datetimes
        datetimes = dates.array(
            data.index[0],
            data.index[-1],
            unit,
            step,
            index=data.index,
            complete=False,
            fixed=True,
        )

    else:
        # dataframe of data
        data = pandas.DataFrame(data).squeeze()

        # array of datetimes
        datetimes = numpy.array([[0, len(data)]])

        # setting step
        step = 1

    if threshold is None:
        τ = 0
    else:
        if isinstance(threshold, (int, float)):
            τ = threshold
        elif isinstance(threshold, str):
            τ = quantile(data.to_numpy(), threshold)

        # applying threshold
        data = data.loc[data.to_numpy() > τ] - τ

    if blocks is None or step > 1:
        index = pandas.Index(
            [
                "%s - %s"
                % (
                    datetime[0],
                    datetime[-1],
                )
                for datetime in datetimes
            ],
            name="block",
        )
    else:
        index = pandas.Index(datetimes[:, 0], name="block")

    if blocks is None and datetimes.dtype == "<U4":
        δ = numpy.asarray(data.index.year.unique().shape[0]).reshape(-1, 1)
    else:
        δ = numpy.diff(datetimes.astype(int)) + 1

    if statistics:

        if normalize is None:
            κ = 1
        elif isinstance(normalize, (int, float)):
            κ = normalize
        elif normalize == "subset":
            κ = numpy.nan
        elif normalize == "dataset":
            κ = 1 / data.dropna().mean()
        elif normalize == "maxima":
            κ = 1 / data.resample(maxima).max().dropna().mean()

        # vectorizing threshold value
        τ = numpy.full((datetimes.shape[0], 1), τ)

        # vectorizing normalization coefficient
        κ = numpy.full((datetimes.shape[0], 1), κ)

        # initializing mean vector
        μ = numpy.full((datetimes.shape[0], 1), numpy.nan)

        # initializing size vector
        λ = numpy.full((datetimes.shape[0], 1), numpy.nan)

        if function is not None:
            x = numpy.full((datetimes.shape[0], len(outputs)), numpy.nan)

        for i, datetime in enumerate(datetimes):

            if maxima is None:
                subset = (
                    data.loc[datetime[0] : datetime[-1]].dropna().to_numpy()
                )
            else:
                subset = (
                    data.loc[datetime[0] : datetime[-1]]
                    .resample(maxima)
                    .max()
                    .dropna()
                    .to_numpy()
                )

            if normalize == "subset":
                κ[i] = 1 / subset.mean()

            # normalizing values
            subset = subset * κ[i]

            # mean
            μ[i] = subset.mean()

            # size
            λ[i] = subset.size

            if lenght > λ[i, 0]:
                continue

            if isinstance(delta, (int, float)) and numpy.any(
                numpy.abs(numpy.subtract.outer(subset, subset))[
                    numpy.triu_indices(λ[i, 0], 1)
                ]
                <= delta
            ):
                continue

            if function is not None:
                x[i, :] = function(*arguments, data=subset, **keywords)

        if dataframe:
            return pandas.DataFrame(
                index=index,
                columns=["δ", "τ", "κ", "μ", "λ"]
                + (outputs if function is not None else []),
                data=numpy.hstack(
                    [δ, τ, κ, μ, λ] + ([x] if function is not None else [])
                ),
            )
        else:
            return (index, δ, τ, κ, μ, λ) + (
                (x,) if function is not None else ()
            )

    else:
        if dataframe:
            return pandas.DataFrame(index=index, columns=["δ"], data=δ)
        else:
            return index, δ
