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

import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def quantile(
    F,
    function,
    x_0=1e-1,
    threshold=0,
    epsilon=1e-12,
    delta=1e-1,
    x_lw=1e-4,
    x_up=1e4,
    dataframe=True,
    c_F="F",
    c_x="value",
    c_f="flag",
    level=0,
):
    """optimization of quantile(s)

    F          -> non-exceedance cumulative probability;
    function   -> objective function;
    x_0        -> first value to be given to solver;
    threshold  -> threshold for exceedances;
    epsilon    -> convergence value;
    delta      -> steps factor;
    x_lw       -> lower boundary;
    x_up       -> upper boundary;
    dataframe  -> return stats dataframe flag;
    c_F        -> non-exceedance cumulative probability column label;
    c_x        -> quantiles column label;
    c_f        -> flags column label;
    recurrence- > return recurrence time flag;
    level      -> logging level.

    """

    # size
    size = F.size

    # sorting
    F = numpy.sort(F)

    # initializing arrays
    x = numpy.full(size, numpy.nan)
    flags = numpy.full(size, numpy.nan)
    errors = numpy.full(size, numpy.nan)
    differences = numpy.full(size, numpy.nan)

    # porgress counter
    progress = 1

    if numpy.isnan(x_0) or x_0 < 0:
        x_0 = x_lw

    while True:
        try:
            fun = lambda x: function(x, F[-1])
            results = scipy.optimize.fsolve(fun, x_0, maxfev=0, full_output=1)
            errors[-1] = results[1]["fvec"]
            differences[-1] = x_0 - results[0]

        except:
            utils.printing(
                "quantile:",
                progress,
                "of",
                size,
                "encountered value overflow",
                level=level,
            )
            errors[-1] = numpy.inf
            pass

        if (
            abs(errors[-1]) < epsilon
            and abs(differences[-1]) > epsilon
            and results[0] > 0
        ):
            utils.printing(
                "quantile:",
                progress,
                "of",
                size,
                "optimization completed",
                level=level,
            )
            progress = progress + 1
            x[-1] = results[0]
            flags[-1] = 1
            break

        else:
            if x_0 <= x_up:
                utils.printing(
                    "quantile:",
                    progress,
                    "of",
                    size,
                    "changing initial value",
                    level=level,
                )
                x_0 = x_0 * (1 + delta)
                flags[-1] = 0
                pass

            else:
                utils.printing(
                    "quantile:",
                    progress,
                    "of",
                    size,
                    "quitting without solution",
                    level=level,
                )
                progress = progress + 1
                flags[-1] = -1
                break

    if flags[-1] != 1:
        utils.printing("quantile: aborting subsequent values", level=level)
    else:
        for i in range(size - 2, -1, -1):
            # initial value
            x_0 = x[i + 1]

            if numpy.isnan(x_0) or numpy.isinf(x_0) or x_0 < 0:
                x_0 = x.min()

            if numpy.isnan(x_0) or numpy.isinf(x_0) or x_0 < 0:
                x_0 = x_up

            while True:
                try:
                    fun = lambda x: function(x, F[i])
                    results = scipy.optimize.fsolve(
                        fun, x_0, maxfev=0, full_output=1
                    )
                    errors[i] = results[1]["fvec"]
                    differences[i] = x_0 - results[0]

                except:
                    utils.printing(
                        "quantile:",
                        progress,
                        "of",
                        size,
                        "encountered value overflow",
                        level=level,
                    )
                    errors[i] = numpy.inf
                    pass

                if (
                    abs(errors[i]) < epsilon
                    and abs(differences[i]) > epsilon
                    and results[0] > 0
                ):
                    utils.printing(
                        "quantile:",
                        progress,
                        "of",
                        size,
                        "optimization completed",
                        level=level,
                    )
                    progress = progress + 1
                    x[i] = results[0]
                    flags[i] = 1
                    break

                else:
                    if x_0 >= x_lw:
                        utils.printing(
                            "quantile:",
                            progress,
                            "of",
                            size,
                            "changing initial value",
                            level=level,
                        )
                        x_0 = x_0 * (1 / (1 + delta))
                        flags[i] = 0
                        pass

                    else:
                        utils.printing(
                            "quantile:",
                            progress,
                            "of",
                            size,
                            "quitting without solution",
                            level=level,
                        )
                        progress = progress + 1
                        flags[i] = -1
                        break

        if size == int(sum(flags)):
            utils.printing("quantile: curve success", level=level)
        else:
            utils.printing("quantile: curve failed", level=level)

    if isinstance(threshold, (int, float)):
        x = x + threshold

    if dataframe:
        return pandas.DataFrame(
            data=zip(F, x, *(flags,) if flags is not None else ()),
            columns=[c_F, c_x] + [c_f] if flags is not None else [],
        )
    else:
        return x


def minimizer(
    function,
    *args,
    par=None,
    data=None,
    method=None,
    bounds=None,
    tol=None,
    level=0,
    **keywords
):
    """optimization of parameter(s)

    function -> objective function;
    par      -> parameter set of the distribution;
    args     -> optional function arguments;
    data     -> input data;
    method   -> optimization method;
    bounds   -> optimization bounds;
    tol      -> optimization tolerance;
    level    -> logging level;
    keywords -> optimization keywords.

    """

    if not isinstance(par, list):
        par = [par]
    if not isinstance(data, list):
        data = [data]

    # number of inputs
    n = max(len(par), len(data))

    if len(par) != n:
        par = par * n
    if len(data) != n:
        data = data * n

    # processing results
    results = [
        scipy.optimize.minimize(
            function,
            i,
            (j, *args),
            method=method,
            bounds=bounds,
            tol=tol,
            **keywords,
        )
        for i, j in zip(par, data)
    ]

    # filtering results
    results = [
        j.x if j.success else numpy.full(len(par), numpy.nan)
        for i, j in zip(par, results)
    ]

    if len(results) == 1:
        return results[0].reshape((1, -1))
    else:
        return numpy.vstack(results)
