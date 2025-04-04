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


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import utils
import validation

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def gev(
    empiricals,
    values,
    indexer,
    matrix,
    maxima="y",
    threshold=None,
    normalize="subset",
    coefficients=None,
    function=["dists", "quantile"],
    τ=0,
    κ=1,
    **keywords,
):
    """gev function"""

    # validation estimator
    n, par = validation.estimator(
        values,
        indexer,
        matrix,
        maxima=maxima,
        threshold=threshold,
        normalize=normalize,
        distribution="gev",
        coefficients=coefficients,
    )

    return validation.evaluator(
        empiricals,
        "gev",
        par,
        axis=0,
        function=function,
        threshold=τ * κ,
        **keywords,
    )


def mev(
    empiricals,
    n,
    array,
    indexer,
    parameters=None,
    distribution="weibull",
    coefficients=None,
    function=["mev", "quantile"],
    v=0,
    τ=0,
    κ=1,
    **keywords,
):
    """mev function"""

    if parameters is None:
        n, par = validation.extractor(
            n,
            array,
            indexer,
            regional=True,
            distribution=distribution,
            coefficients=coefficients,
        )
    else:
        n, par = validation.extractor(
            n,
            array,
            indexer,
            regional=None,
            distribution=None,
            coefficients=coefficients,
        )

    return validation.evaluator(
        empiricals,
        distribution,
        par,
        n,
        axis=-1,
        function=function,
        v_0=v * κ,
        threshold=τ * κ,
        **keywords,
    )


def smev(
    empiricals,
    values,
    indexer,
    matrix,
    maxima=None,
    threshold=None,
    normalize="dataset",
    distribution="weibull",
    coefficients=None,
    function=["mev", "quantile"],
    v=0,
    τ=0,
    κ=1,
    **keywords,
):
    """smev function"""

    # validation estimator
    n, par = validation.estimator(
        values,
        indexer,
        matrix,
        maxima=maxima,
        threshold=threshold,
        normalize=normalize,
        distribution=distribution,
        coefficients=coefficients,
    )

    return validation.evaluator(
        empiricals,
        distribution,
        par,
        n,
        axis=0,
        function=function,
        v_0=v * κ,
        threshold=τ * κ,
        **keywords,
    )


def function(name, *arguments, key=None, separator=" ", **keywords):
    """cross-validator function"""

    # computing quantiles
    output = globals()[name](*arguments, **keywords)

    if isinstance(output, tuple):
        dataframe = output[0]
        output = output[1:]
    else:
        dataframe = output
        output = ()

    if "κ" in keywords:
        dataframe.loc[:, "model"] = dataframe.loc[:, "model"].div(
            keywords["κ"]
        )

    # setting distribution
    dataframe.loc[:, "distribution"] = utils.concatenating(
        name, key, separator=separator
    )

    # setting index
    dataframe = dataframe.reset_index().set_index(
        ["extraction", "distribution"]
    )

    if len(output) == 0:
        return dataframe
    else:
        return dataframe, *output
