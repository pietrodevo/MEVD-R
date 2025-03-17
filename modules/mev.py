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

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import dists
import optimization
import stats
import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def fit(
    data,
    distribution,
    index=None,
    column=None,
    blocks="y",
    maxima=None,
    threshold=None,
    normalize=None,
    parameters=2,
    estimator="lm",
    dataframe=None,
    *args,
    **kwargs,
):
    """variable frequency curve

    data         -> input dataframe/array/tuple/list;
    distribution -> selected distribution name;
    index        -> dataframe index column;
    column       -> dataframe values column;
    blocks       -> optional temporal blocks;
    maxima       -> optional temporal maxima;
    threshold    -> optional threshold value/quantile/percentile;
    normalize    -> optional normalization value/mode;
    parameters   -> number of distribution parameters;
    estimator    -> estimation method (lm/nll);
    dataframe    -> flagging dataframe output;
    args         -> optional arguments for the distribution function;
    kwargs       -> optional keyword arguments for the distribution function.

    """

    # computing parameters
    dtf = stats.blocks(
        data,
        function=dists.parameters,
        index=index,
        column=column,
        blocks=blocks,
        maxima=maxima,
        threshold=threshold,
        normalize=normalize,
        outputs=parameters,
        arguments=(distribution, *args),
        keywords={
            "estimator": estimator,
            **kwargs,
        },
    )

    if dataframe is None:
        return (
            dtf.iloc[:, -parameters:].to_numpy(),
            (dtf.loc[:, "λ"] / dtf.loc[:, "δ"]).to_numpy(),
        )
    elif dataframe is True:
        return dataframe
    else:
        return dtf.loc[:, dataframe]


def function(
    x,
    distribution,
    par,
    n,
    *args,
    threshold=None,
    reshape=True,
    drop=True,
    **kwargs,
):
    """MEV distribution function

    x            -> variable value;
    distribution -> designed distribution function;
    par          -> parameter set of the distribution;
    n            -> number of exceedences;
    args         -> optional arguments for the distribution function;
    threshold    -> threshold for exceedances;
    reshape      -> dimensions reshaping flag;
    drop         -> nans dropping flag;
    kwargs       -> optional keyword arguments for the distribution function.

    axis 1 -> dimension of blocks;
    axis 2 -> dimension of parameters.

    """

    if type(x).__name__ != "ndarray":
        x = numpy.array(x).squeeze()

    if reshape:
        if isinstance(par, (int, float)):
            par = numpy.array([par])
        if par.ndim <= 1:
            par = par.reshape(1, -1)

        if isinstance(n, (int, float)):
            n = numpy.array([n])
        if n.ndim <= 1:
            n = n.reshape(1, -1)

    if isinstance(threshold, (int, float)):
        x = x + threshold

    if drop:
        # indexing non-nan values
        index = numpy.invert(
            numpy.all(numpy.isnan(par), axis=1)
            | numpy.all(numpy.isnan(n), axis=1)
        )

        # filtering arrays
        par, n = par[index, :], n[index, :]

    # number of blocks
    N = max(par.shape[0], n.shape[1])

    return (
        1
        / N
        * numpy.nansum(
            dists.cdf(x, distribution, par, *args, reshape=False, **kwargs)
            ** n[:, 0],
        )
    )


def quantile(
    F,
    distribution,
    par,
    n,
    *args,
    v_0=1e-1,
    threshold=0,
    epsilon=1e-12,
    delta=1,
    x_lw=1e-3,
    x_up=1e3,
    dataframe=True,
    c_F="F",
    c_x="value",
    c_f="flag",
    recurrence=None,
    equation="function",
    curve=None,
    level=1,
    **kwargs,
):
    """quantile function

    F            -> non-exceedance cumulative probability values;
    distribution -> designed distribution function;
    par          -> parameter set of the distribution;
    n            -> number of exceed days per reference period;
    args         -> optional arguments for the distribution function;
    v_0          -> first value to be given to solver;
    threshold    -> threshold for exceedances;
    epsilon      -> convergence value;
    delta        -> steps factor;
    x_lw         -> lower boundary;
    x_up         -> upper boundary;
    dataframe    -> return stats dataframe flag;
    c_F          -> non-exceedance cumulative probability column label;
    c_x          -> quantiles column label;
    c_f          -> flags column label;
    recurrence   -> return recurrence time flag;
    equation     -> MEV function equation;
    curve        -> curve generation flag or non-exceedance cumulative probability values;
    level        -> logging level;
    kwargs       -> optional keyword arguments for the distribution function.

    """

    if F is None:
        F = numpy.array([])
    elif isinstance(F, (list, tuple)):
        F = numpy.array(F)
    elif type(F).__name__ == "Series":
        F = F.to_numpy()
    elif type(F).__name__ == "DataFrame":
        F = F.loc[:, "F"].to_numpy()

    if not isinstance(threshold, (int, float)):
        threshold = 0

    if curve is True:
        F = utils.ranges(
            [0.001, 0.050, 0.950, 0.999],
            [0.001, 0.010, 0.001],
            add=F,
            sort=True,
        )
    elif curve is not None:
        F = numpy.sort(
            numpy.concat(
                [F, [curve] if isinstance(curve, (int, float)) else curve]
            )
        )

    # objective
    φ = (
        lambda x, F: globals()[equation](
            x,
            distribution,
            par,
            n,
            *args,
            **kwargs,
        )
        - F
    )

    return optimization.quantile(
        F,
        φ,
        v_0,
        threshold,
        epsilon,
        delta,
        x_lw,
        x_up,
        dataframe,
        c_F,
        c_x,
        c_f,
        level,
    )
