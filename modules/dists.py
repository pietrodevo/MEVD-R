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

import optimization
import stats
import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def extractor(data, *L, array=None, method="pwm", order=3):
    """moments(s) extract function

    data   -> input values if linear moments are not provided;
    L      -> linear moment(s);
    array  -> optional linear moments array;
    method -> estimation method;
    order  -> estimation order.

    """

    if data is not None:
        return stats.lm(data, method, order, array=False)
    elif array is not None:
        if array.ndim == 1:
            array = numpy.expand_dims(array, axis=0)
        return (array[:, i].reshape(-1, 1) for i in range(order))
    else:
        return (
            numpy.array([i]) if isinstance(i, (int, float)) else i for i in L
        )


def recurrence(data, output="RP", c_F="F", c_RP="RP"):
    """recurrence function

    data   -> input data value/array/dataframe;
    output -> selected RP/F output;
    c_F    -> non-exceedance cumulative probability column label;
    c_RP   -> return period column label.

    """

    if output == "RP":
        fun = lambda F: 1 / (1 - F)
        columns = [c_RP, c_F]
    elif output == "F":
        fun = lambda RP: 1 - 1 / RP
        columns = [c_F, c_RP]
    else:
        return

    if type(data).__name__ == "DataFrame":
        return data.assign(**{columns[0]: fun(data.loc[:, columns[1]])})

    else:
        return fun(numpy.array(data))


def probability(
    x,
    distribution,
    par,
    *args,
    points=1000,
    function="pdf",
    c_x="value",
    c_p="p",
    threshold=0,
    dataframe=True,
    **kwargs,
):
    """distribution probability function

    x            -> input values;
    distribution -> selected distribution name;
    par          -> parameter set of the distribution;
    args         -> optional arguments for the distribution function;
    points       -> probability generation points;
    function     -> probability generation function;
    c_p          -> probability column label;
    c_x          -> values column label;
    dataframe    -> return stats dataframe flag;
    kwargs       -> optional keywords for the distribution function.

    """

    if x is None:
        x = numpy.array([])
    elif isinstance(x, (list, tuple)):
        x = numpy.array(x)
    elif type(x).__name__ == "Series" or x.shape[1] == 1:
        x = x.to_numpy()
    elif type(x).__name__ == "DataFrame":
        x = x.loc[:, "x"].to_numpy()

    # generating points
    x = numpy.linspace(x.min(), x.max(), points)

    # quantiles computing
    p = globals()[function + "_" + distribution](
        x, par, *args, **kwargs
    ).reshape(-1)

    if dataframe:
        return pandas.DataFrame(
            data=zip(p, x),
            columns=[c_p, c_x],
        )
    else:
        return p


def quantile(
    F,
    distribution,
    par,
    *args,
    c_x="value",
    c_F="F",
    threshold=0,
    dataframe=True,
    curve=None,
    **kwargs,
):
    """distribution quantile function

    F            -> non-exceedance cumulative probability values;
    distribution -> selected distribution name;
    par          -> parameter set of the distribution;
    args         -> optional arguments for the distribution function;
    c_x          -> quantiles column label;
    c_F          -> non-exceedance cumulative probability column label;
    threshold    -> threshold for exceedances;
    dataframe    -> return stats dataframe flag;
    curve        -> curve generation flag or non-exceedance cumulative probability values;
    kwargs       -> optional keywords for the distribution function.

    """

    if F is None:
        F = numpy.array([])
    elif isinstance(F, (list, tuple)):
        F = numpy.array(F)
    elif type(F).__name__ == "Series" or F.shape[1] == 1:
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

    # quantiles computing
    x = globals()["quantile" + "_" + distribution](
        F, par, *args, **kwargs
    ).reshape(-1)

    if isinstance(threshold, (int, float)):
        x = x + threshold

    if dataframe:
        return pandas.DataFrame(
            data=zip(x, F),
            columns=[c_x, c_F],
        )
    else:
        return x


def quantile_gumbel(F, par, reshape=3):
    """quantile function values for gumbel

    F       -> non-exceedance cumulative probability values;
    par     -> parameter set of the distribution;
       [0]  -> scale (α);
       [1]  -> location (ξ);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, F = utils.reshaper(par, F, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.gumbel_r.ppf(F, scale=par[0], loc=par[1])
    else:
        return scipy.stats.gumbel_r.ppf(
            F, scale=par[:, 0, ...], loc=par[:, 1, ...]
        )


def quantile_gev(F, par, reshape=3):
    """quantile function values for generalized extreme value

    F       -> non-exceedance cumulative probability values;
    par     -> parameter set of the distribution;
       [0]  -> shape (ĸ);
       [1]  -> scale (α);
       [2]  -> location (ξ);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, F = utils.reshaper(par, F, n=3, s=reshape)

    if par.ndim == 1:
        return scipy.stats.genextreme.ppf(F, par[0], scale=par[1], loc=par[2])
    else:
        return scipy.stats.genextreme.ppf(
            F, par[:, 0, ...], scale=par[:, 1, ...], loc=par[:, 2, ...]
        )


def quantile_weibull(F, par, reshape=3):
    """quantile function values for weibull

    F       -> non-exceedance cumulative probability values;
    par     -> parameter set of the distribution;
       [0]  -> shape (ĸ);
       [1]  -> scale (α);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, F = utils.reshaper(par, F, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.weibull_min.ppf(F, par[0], scale=par[1], loc=0)
    else:
        return scipy.stats.weibull_min.ppf(
            F, par[:, 0, ...], scale=par[:, 1, ...], loc=0
        )


def quantile_gamma(F, par, reshape=3):
    """quantile function values for gamma

    F       -> non-exceedance cumulative probability values;
    par     -> parameter set of the distribution;
       [0]  -> shape (α);
       [1]  -> scale (β);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, F = utils.reshaper(par, F, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.gamma.ppf(F, par[0], scale=par[1], loc=0)
    else:
        return scipy.stats.gamma.ppf(
            F, par[:, 0, ...], scale=par[:, 1, ...], loc=0
        )


def pdf(x, distribution, par, *args, threshold=0, **kwargs):
    """distribution probability density function

    x            -> variable value(s);
    distribution -> selected distribution name;
    par          -> parameter set of the distribution;
    args         -> optional arguments for the distribution function;
    threshold    -> threshold for exceedances;
    kwargs       -> optional keyword arguments for the distribution function.

    """

    if isinstance(threshold, (int, float)):
        x = x + threshold

    return globals()["pdf" + "_" + distribution](x, par, *args, **kwargs)


def pdf_gumbel(x, par, reshape=3):
    """probability density function for gumbel

    x       -> variable value(s);
    par     -> parameter set of the distribution;
       [0]  -> scale (α);
       [1]  -> location (ξ);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=3, s=reshape)

    if par.ndim == 1:
        return scipy.stats.gumbel_r.pdf(x, scale=par[0], loc=par[1])
    else:
        return scipy.stats.gumbel_r.pdf(
            x, scale=par[:, 0, ...], loc=par[:, 1, ...]
        )


def pdf_gev(x, par, reshape=3):
    """probability density function for generalized extreme value

    par     -> parameters array;
       [0]  -> shape (ĸ);
       [1]  -> scale (α);
       [2]  -> location (ξ);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=3, s=reshape)

    if par.ndim == 1:
        return scipy.stats.genextreme.pdf(x, par[0], scale=par[1], loc=par[2])
    else:
        return scipy.stats.genextreme.pdf(
            x, par[:, 0, ...], scale=par[:, 1, ...], loc=par[:, 2, ...]
        )


def pdf_weibull(x, par, reshape=3):
    """probability density function for weibull

    x       -> variable value(s);
    par     -> parameter set of the distribution;
       [0]  -> shape (ĸ);
       [1]  -> scale (α);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.weibull_min.pdf(x, par[0], scale=par[1], loc=0)
    else:
        return scipy.stats.weibull_min.pdf(
            x, par[:, 0, ...], scale=par[:, 1, ...], loc=0
        )


def pdf_gamma(x, par, reshape=3):
    """probability density function for gamma

    x       -> variable value(s);
    par     -> parameter set of the distribution;
       [0]  -> shape (α);
       [1]  -> scale (β);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.gamma.pdf(x, par[0], scale=par[1], loc=0)
    else:
        return scipy.stats.gamma.pdf(
            x, par[:, 0, ...], scale=par[:, 1, ...], loc=0
        )


def cdf(x, distribution, par, *args, threshold=0, **kwargs):
    """distribution cumulative function

    x            -> variable value(s);
    distribution -> selected distribution name;
    par          -> parameter set of the distribution;
    args         -> optional arguments for the distribution function;
    threshold    -> threshold for exceedances;
    kwargs       -> optional keyword arguments for the distribution function.

    """

    if isinstance(threshold, (int, float)):
        x = x - threshold

    return globals()["cdf" + "_" + distribution](x, par, *args, **kwargs)


def cdf_gumbel(x, par, reshape=3):
    """cumulative distribution function for gumbel

    x       -> variable value(s);
    par     -> parameter set of the distribution;
       [0]  -> scale (α);
       [1]  -> location (ξ);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.gumbel_r.cdf(x, scale=par[0], loc=par[1])
    else:
        return scipy.stats.gumbel_r.cdf(
            x, scale=par[:, 0, ...], loc=par[:, 1, ...]
        )


def cdf_gev(x, par, reshape=3):
    """cumulative distribution function for generalized extreme value

    x       -> variable value(s);
    par     -> parameter set of the distribution;
       [0]  -> shape (ĸ);
       [1]  -> scale (α);
       [2]  -> location (ξ);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=3, s=reshape)

    if par.ndim == 1:
        return scipy.stats.genextreme.cdf(x, par[0], scale=par[1], loc=par[2])
    else:
        return scipy.stats.genextreme.cdf(
            x, par[:, 0, ...], scale=par[:, 1, ...], loc=par[:, 2, ...]
        )


def cdf_weibull(x, par, reshape=3):
    """cumulative distribution function for weibull

    x       -> variable value(s);
    par     -> parameter set of the distribution;
       [0]  -> shape (ĸ);
       [1]  -> scale (α);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.weibull_min.cdf(x, par[0], scale=par[1], loc=0)
    else:
        return scipy.stats.weibull_min.cdf(
            x, par[:, 0, ...], scale=par[:, 1, ...], loc=0
        )


def cdf_gamma(x, par, reshape=3):
    """cumulative distribution function for gamma

    x       -> variable value(s);
    par     -> parameter set of the distribution;
       [0]  -> shape (α);
       [1]  -> scale (β);
    reshape -> dimensions reshape.

    """

    if reshape:
        par, x = utils.reshaper(par, x, n=2, s=reshape)

    if par.ndim == 1:
        return scipy.stats.gamma.cdf(x, par[0], scale=par[1], loc=0)
    else:
        return scipy.stats.gamma.cdf(
            x, par[:, 0, ...], scale=par[:, 1, ...], loc=0
        )


def parameters(
    distribution=None, data=None, estimator="lm", drop=True, **keywords
):
    """distribution parameters estimated via linear moments (lm) or negative log likelihood (nll)

    distribution -> selected distribution name;
    data         -> input values if lm are not provided or if using nll;
    estimator    -> estimation method (lm/nll);
    drop         -> invalid values checking flag;
    **keywords   -> optional keyword arguments for the distribution function.

    """

    if type(data).__name__ in ["Series", "DataFrame"]:
        data = data.to_numpy()

    if data is not None and drop:
        data = data[~numpy.isnan(data)]

    return globals()[
        "parameters" + "_" + distribution.lower() + "_" + estimator.lower()
    ](data, **keywords)


def parameters_gumbel_lm(data=None, L1=None, L2=None, array=None):
    """parameters of the gumbel distribution estimated via linear moments

    data  -> input values if linear moments are not provided;
    L1    -> order 1 linear moment(s);
    L2    -> order 2 linear moment(s);
    array -> optional linear moments array.

    """

    # extracting arguments
    L1, L2 = extractor(data, L1, L2, array=array, order=2)

    # estimate the scale (α) parameter(s) of the distribution
    scale = L2 / numpy.log(2)

    # estimate the location (ξ) parameter(s) of the distribution
    location = L1 - 0.577215 * scale

    if scale.shape[0] == 1 or location.shape[0] == 1:
        return numpy.array([scale, location]).squeeze()
    else:
        return numpy.hstack([scale, location])


def parameters_gumbel_nll(
    data=None,
    par=None,
    method="SLSQP",
    bounds=[(1e-5, None), (None, None)],
    tol=1e-9,
    **keywords,
):
    """parameters of the gumbel distribution estimated via negative log likelihood

    data     -> input values;
    par      -> optional initial guess parameter set of the distribution;
       [0]   -> scale (α);
       [1]   -> location (ξ);
    method   -> optimization method;
    bounds   -> optimization bounds;
    tol      -> optimization tolerance;
    keywords -> optimization keywords.

    """

    # aligning arguments
    data, par = utils.aligner(data, par)

    # optimized parameters
    par = optimization.minimizer(
        nll_gumbel,
        par=[parameters_gumbel_lm(i) for i, j in zip(data, par) if j is None],
        data=data,
        method=method,
        bounds=bounds,
        tol=tol,
        level=1,
        **keywords,
    )

    if par.shape[0] == 1:
        return par.squeeze()
    else:
        return par


def parameters_gev_lm(
    data=None, L1=None, L2=None, L3=None, array=None, tail=None
):
    """parameters of the generalized extreme value distribution estimated via linear moments

    data  -> input values if linear moments are not provided;
    L1    -> order 1 linear moment(s);
    L2    -> order 2 linear moment(s);
    L3    -> order 3 linear moment(s);
    array -> optional linear moments array;
    tail  -> tail adjust flag.

    """

    # extracting arguments
    L1, L2, L3 = extractor(data, L1, L2, array=array, order=3)

    # costant values
    c = 2 / (3 + L3 / L2) - numpy.log(2) / numpy.log(3)
    k = 7.8590 * c + 2.9554 * c**2

    # estimate the shape (ĸ) parameter(s) of the distribution
    shape = k
    # estimate the scale (α) parameter(s) of the distribution
    scale = (L2 * k) / ((1 - 2 ** (-k)) * scipy.special.gamma(1 + k))
    # estimate the location (ξ) parameter(s) of the distribution
    location = L1 + scale * (scipy.special.gamma(1 + k) - 1) / k

    if tail:
        # indexing negative values
        index = numpy.where(shape > 0)[0]

        # fix the shape (ĸ) parameter(s) of the distribution
        shape[index] = 0

        if L1.shape[0] == 1 or L2.shape[0] == 1:
            scale[index], location[index] = parameters_gumbel_lm(
                array=numpy.hstack([L1, L2])
            )
        else:
            scale[index], location[index] = numpy.split(
                parameters_gumbel_lm(array=numpy.hstack([L1, L2]))[index, :],
                2,
                axis=1,
            )

    if shape.shape[0] == 1 or scale.shape[0] == 1 or location.shape[0] == 1:
        return numpy.array([shape, scale, location]).squeeze()
    else:
        return numpy.hstack([shape, scale, location])


def parameters_gev_nll(
    data=None,
    par=None,
    bounds=[(None, None), (1e-5, None), (None, None)],
    tol=1e-9,
    **keywords,
):
    """parameters of the generalized extreme value distribution estimated via negative log likelihood

    data     -> input values;
    par      -> optional initial guess parameter set of the distribution;
       [0]   -> shape (ĸ);
       [1]   -> scale (α);
       [2]   -> location (ξ);
    bounds   -> optimization bounds;
    tol      -> optimization tolerance;
    keywords -> optimization keywords.

    """

    # aligning arguments
    data, par = utils.aligner(data, par)

    # optimized parameters
    par = optimization.minimizer(
        nll_gev,
        par=[parameters_gev_lm(i) for i, j in zip(data, par) if j is None],
        data=data,
        bounds=bounds,
        tol=tol,
        level=1,
        **keywords,
    )

    if par.shape[0] == 1:
        return par.squeeze()
    else:
        return par


def parameters_weibull_lm(data=None, L1=None, L2=None, array=None):
    """parameters of the weibull distribution estimated via linear moments

    data  -> input values if linear moments are not provided;
    L1    -> order 1 linear moment(s);
    L2    -> order 2 linear moment(s);
    array -> optional linear moments array.

    """

    # extracting arguments
    L1, L2 = extractor(data, L1, L2, array=array, order=2)

    # estimate the shape (ĸ) parameter(s) of the distribution
    shape = -numpy.log(2) / numpy.log(1 - L2 / L1)

    # estimate the scale (α) parameter(s) of the distribution
    scale = L1 / scipy.special.gamma(1 / shape + 1)

    if shape.shape[0] == 1 or scale.shape[0] == 1:
        return numpy.array([shape, scale]).squeeze()
    else:
        return numpy.hstack([shape, scale])


def parameters_weibull_nll(
    data=None,
    par=None,
    method="SLSQP",
    bounds=[(1e-5, None), (1e-5, None)],
    tol=1e-9,
    **keywords,
):
    """parameters of the weibull distribution estimated via negative log likelihood

    data     -> input values;
    par      -> optional initial guess parameter set of the distribution;
       [0]   -> shape (ĸ);
       [1]   -> scale (α);
    method   -> optimization method;
    bounds   -> optimization bounds;
    keywords -> optimization keywords.

    """

    # aligning arguments
    data, par = utils.aligner(data, par)

    # optimized values
    par = optimization.minimizer(
        nll_weibull,
        par=[parameters_weibull_lm(i) for i, j in zip(data, par) if j is None],
        data=data,
        method=method,
        bounds=bounds,
        tol=tol,
        level=1,
        **keywords,
    )

    if par.shape[0] == 1:
        return par.squeeze()
    else:
        return par


def parameters_gamma_lm(data=None, L1=None, L2=None, array=None):
    """parameters of the gamma distribution estimated via linear moments

    data  -> input values if linear moments are not provided;
    L1    -> order 1 linear moment(s);
    L2    -> order 2 linear moment(s);
    array -> optional linear moments array.

    """

    # extracting arguments
    L1, L2 = extractor(data, L1, L2, array=array, order=2)

    # initialize
    z = numpy.ones((min(L1.shape[0], L1.shape[0]), 1)) * numpy.nan
    shape = numpy.ones((L1.shape[0], 1)) * numpy.nan

    # indexing
    index_1 = (L2 / L1 > 0) & (L2 / L1 < 0.5)
    index_2 = (L2 / L1 >= 0.5) & (L2 / L1 < 1)

    # costant value
    z[index_1] = numpy.pi * (L2[index_1] / L1[index_1]) ** 2
    z[index_2] = 1 - (L2[index_2] / L1[index_2])

    # estimate the shape (α) parameter(s) of the distribution
    shape[index_1] = (1 - 0.3080 * z[index_1]) / (
        z[index_1] - 0.05812 * z[index_1] ** 2 + 0.01765 * z[index_1] ** 3
    )
    shape[index_2] = (0.7213 * z[index_2] - 0.5947 * z[index_2] ** 2) / (
        1 - 2.1817 * z[index_2] + 1.2113 * z[index_2] ** 2
    )

    # estimate the scale (β) parameter(s) of the distribution
    scale = L1 / shape

    if shape.shape[0] == 1 or scale.shape[0] == 1:
        return numpy.array([shape, scale]).squeeze()
    else:
        return numpy.hstack([shape, scale])


def parameters_gamma_nll(
    data=None,
    par=None,
    method="SLSQP",
    bounds=[(1e-5, None), (1e-5, None)],
    tol=1e-9,
    **keywords,
):
    """parameters of the gamma distribution estimated via negative log likelihood

    data     -> input values;
    par      -> optional initial guess parameter set of the distribution;
       [0]   -> shape (α);
       [1]   -> scale (β);
    method   -> optimization method;
    bounds   -> optimization bounds;
    tol      -> optimization tolerance;
    keywords -> optimization keywords.

    """

    # aligning arguments
    data, par = utils.aligner(data, par)

    # optimized parameters
    par = optimization.minimizer(
        nll_gamma,
        par=[parameters_gamma_lm(i) for i, j in zip(data, par) if j is None],
        data=data,
        method=method,
        bounds=bounds,
        tol=tol,
        level=1,
        **keywords,
    )

    if par.shape[0] == 1:
        return par.squeeze()
    else:
        return par


def nll(distribution=None, par=None, data=None):
    """negative log likelihood

    distribution -> selected distribution name;
    par          -> parameter set of the distribution;
    data         -> input values.

    """

    return globals()["nll" + "_" + distribution.lower()](par, data)


def nll_gumbel(par=None, data=None):
    """negative log likelihood of the gumbel distribution

    par    -> parameters array;
       [0] -> scale (α);
       [1] -> location (ξ);
    data   -> input values.

    """

    return -numpy.log(pdf_gumbel(data, par, reshape=1)).sum()


def nll_gev(par=None, data=None):
    """negative log likelihood of the generalized extreme value distribution

    par    -> parameters array;
       [0] -> shape (ĸ);
       [1] -> scale (α);
       [2] -> location (ξ);
    data   -> input values;

    """

    return -numpy.log(pdf_gev(data, par, reshape=1)).sum()


def nll_weibull(par=None, data=None):
    """negative log likelihood of the weibull distribution

    par    -> parameters array;
       [0] -> shape (ĸ);
       [1] -> scale (α);
    data   -> input values.

    """

    return -numpy.log(pdf_weibull(data, par, reshape=1)).sum()


def nll_gamma(par=None, data=None):
    """negative log likelihood of the gamma distribution

    par    -> parameters array;
       [0] -> shape (α);
       [1] -> scale (β);
    data   -> input values.

    """

    return -numpy.log(pdf_gamma(data, par, reshape=1)).sum()


def pp(position=None, data=None):
    """plotting position function

    position -> selected position;
    data     -> input values.

    """

    if not type(data).__name__ != "ndarray":
        data = numpy.array(data)

    return globals()["pp" + "_" + position.lower()](data)


def pp_weibull(data=None):
    """weibull plotting position

    data -> input values.

    """

    return numpy.arange(1, data.size + 1, 1) / (data.size + 1)


def pp_median(data=None):
    """median plotting position

    data -> input values.

    """

    return numpy.arange(0.6825, data.size + 0.6825, 1) / (data.size + 0.365)


def pp_apl(data=None):
    """approximated partial likelihood plotting position

    data  > input values.

    """

    return numpy.arange(0.65, data.size + 0.65, 1) / data.size


def pp_bloom(data=None):
    """bloom plotting position

    data -> input values.

    """

    return numpy.arange(0.625, data.size + 0.625, 1) / (data.size + 0.25)


def pp_cunnane(data=None):
    """cunnane plotting position

    data -> input values.

    """

    return numpy.arange(0.6, data.size + 0.6, 1) / (data.size + 0.2)


def pp_gringorten(data=None):
    """gringorten plotting position

    data -> input values.

    """

    return numpy.arange(0.52, data.size + 0.52, 1) / (data.size + 0.12)


def pp_hazen(data=None):
    """hazen plotting position

    data -> input values.

    """

    return numpy.arange(0.5, data.size + 0.5, 1) / data.size


def empirical(
    x,
    position="weibull",
    dataframe=True,
    c_F="F",
    c_x="value",
    sequence=True,
):
    """empirical cumulative probability

    x           -> input value/array/dataframe;
    position    -> plotting position;
    dataframe   -> return stats dataframe flag;
    c_F         -> non-exceedance cumulative probability column label;
    c_x         -> quantiles column label;
    sequence    -> return original sequence flag.

    """

    if isinstance(x, (list, tuple)):
        x = numpy.array(x)
    elif type(x).__name__ == "Series" or x.shape[1] == 1:
        x = x.to_numpy()
    elif type(x).__name__ == "DataFrame":
        x = x.loc[:, "x"]

    # unsorted data
    u = x[~numpy.isnan(x)]

    # sorting values
    x = numpy.sort(u)

    # plotting positions
    F = pp(position, x)

    if sequence:
        # indexing
        index = numpy.searchsorted(x, u)

        # reordering
        x, F = x[index], F[index]

    if dataframe:
        return pandas.DataFrame(
            data=zip(F, x),
            columns=[c_F, c_x],
        )
    else:
        return F


def evaluation(
    x,
    y,
    relate="F",
    source="value",
    destination="model",
    tolerance=None,
    indexer=False,
):
    """distributions evaluation

    x           -> input x value/array/dataframe;
    y           -> input y value/array/dataframe;
    relate      -> column(s) for associations;
    source      -> label in the source dataframe;
    destination -> label in the destination dataframe;
    tolerance   -> matching tolerance flag/value;
    indexer     -> return indexer flag.

    """

    if not isinstance(relate, (list, tuple)):
        relate = [relate] * 2

    if type(x).__name__ == "DataFrame":
        # empirical copy
        out = x.copy()

        if isinstance(tolerance, (int, float)):
            index = (
                x.loc[:, relate[0]]
                .apply(
                    lambda i: (
                        y.loc[(numpy.abs(y[relate[1]] - i) <= tolerance)]
                        .assign(delta=lambda j: numpy.abs(j[relate[1]] - i))
                        .sort_values(by="delta")
                        .index[0]
                        if (numpy.abs(y[relate[1]] - i) <= tolerance).any()
                        else numpy.nan
                    )
                )
                .dropna()
            )
        else:
            index = pandas.Series(
                numpy.abs(
                    numpy.subtract.outer(
                        x.loc[:, relate[0]].to_numpy(),
                        y.loc[:, relate[1]].to_numpy(),
                    )
                ).argmin(axis=1),
                index=x.index,
            )

        # assigning values
        out.loc[index.index, destination] = y.loc[index, source].to_numpy()

    else:
        if type(x).__name__ != "ndarray":
            x = numpy.array(x)
        elif x.ndim == 0:
            x = x.squeeze()

        if isinstance(tolerance, (int, float)):
            index = pandas.Series(
                [
                    (
                        y.loc[(numpy.abs(y[relate[1]] - i) <= tolerance)]
                        .assign(delta=lambda j: numpy.abs(j[relate[1]] - i))
                        .sort_values(by="delta")
                        .index[0]
                        if (numpy.abs(y[relate[1]] - i) <= tolerance).any()
                        else numpy.nan
                    )
                    for i in x
                ]
            )
        else:
            index = pandas.Series(
                numpy.abs(
                    numpy.subtract.outer(
                        x,
                        y.loc[:, relate[1]].to_numpy(),
                    )
                ).argmin(axis=1)
            )

        # initializing output
        out = numpy.full(x.shape, numpy.nan)

        # assigning values
        out[index.dropna().index.to_numpy().astype(int)] = y.iloc[
            index.dropna()
        ].loc[:, source]

        if out.size == 1:
            out = out[0]

    if indexer:
        return index
    else:
        return out
