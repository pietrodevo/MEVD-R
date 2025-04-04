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

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import dates

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def hpd(
    file,
    variables=None,
    frequency="h",
    flags=True,
    nans=None,
    drop=None,
):
    """HPD data file

    file      -> path to file;
    variables -> selected variable(s);
    frequency -> data frequency;
    flags     -> including quality flags.

    """

    # converting frequency
    step, unit = dates.converter(frequency)

    if unit == "h":
        iterator = range(int(24 / step))
        label = "HOUR"
    if unit == "m":
        iterator = range(int(60 * 24 / step))
        label = "MINUTE"
    if unit == "s":
        iterator = range(int(60 * 60 * 24 / step))
        label = "SECOND"

    data_header_names = ["ID", "YEAR", "MONTH", "DAY", "ELEMENT"]

    data_header_col_specs = [(0, 11), (11, 15), (15, 17), (17, 19), (19, 23)]

    data_header_dtypes = {
        "ID": str,
        "YEAR": int,
        "MONTH": int,
        "DAY": int,
        "ELEMENT": str,
    }

    data_col_names = [
        [
            "VALUE" + str(i + 1),
            "MFLAG" + str(i + 1),
            "QFLAG" + str(i + 1),
            "SFLAG" + str(i + 1),
        ]
        for i in iterator
    ]
    data_col_names = sum(data_col_names, [])

    data_replacement_col_names = [
        [
            ("VALUE", i + 1),
            ("MFLAG", i + 1),
            ("QFLAG", i + 1),
            ("SFLAG", i + 1),
        ]
        for i in iterator
    ]
    data_replacement_col_names = sum(data_replacement_col_names, [])
    data_replacement_col_names = pandas.MultiIndex.from_tuples(
        data_replacement_col_names, names=["VARIABLE", label]
    )

    data_col_specs = [
        [
            (23 + i * 9, 28 + i * 9),
            (28 + i * 9, 29 + i * 9),
            (29 + i * 9, 30 + i * 9),
            (30 + i * 9, 31 + i * 9),
        ]
        for i in iterator
    ]
    data_col_specs = sum(data_col_specs, [])

    data_col_dtypes = [
        {
            "VALUE" + str(i + 1): int,
            "MFLAG" + str(i + 1): str,
            "QFLAG" + str(i + 1): str,
            "SFLAG" + str(i + 1): str,
        }
        for i in iterator
    ]

    # updating dtypes
    data_header_dtypes.update(
        {k: v for d in data_col_dtypes for k, v in d.items()}
    )

    # reading
    dataframe = pandas.read_fwf(
        file,
        colspecs=data_header_col_specs + data_col_specs,
        names=data_header_names + data_col_names,
        index_col=data_header_names,
        dtype=data_header_dtypes,
    )

    if variables is not None:
        if not isinstance(variables, list):
            variables = [variables]
        dataframe = dataframe[
            dataframe.index.get_level_values("ELEMENT").isin(variables)
        ]

    # naming
    dataframe.columns = data_replacement_col_names

    if not flags:
        dataframe = dataframe.loc[:, ("VALUE", slice(None))]
        dataframe.columns = dataframe.columns.droplevel("VARIABLE")

    # stacking
    dataframe = dataframe.stack(level=label).unstack(level="ELEMENT")

    # naming
    names = [i[0] if i[0] != "VALUE" else i[1] for i in dataframe.columns]

    # adjustment
    dataframe.columns = dataframe.columns.droplevel("VARIABLE")
    dataframe.columns = names

    # resetting index
    dataframe.reset_index(inplace=True)

    # frequency adjustments
    dataframe.loc[:, label] = dataframe.loc[:, label].multiply(step)

    # setting index
    dataframe.set_index(["ID", "YEAR", "MONTH", "DAY", label], inplace=True)

    # replace the entire index with the date
    dataframe.index = pandas.to_datetime(
        dataframe.reset_index().loc[:, ["YEAR", "MONTH", "DAY", label]],
    )

    # renaming
    dataframe.index.name = "datetime"

    return dataframe


def ghcnd(
    file,
    variables=None,
    frequency="d",
    flags=True,
):
    """GHCND data file

    file      -> path to file;
    variables -> selected variable(s);
    frequency -> data frequency;
    flags     -> including quality flags.

    """

    data_header_names = ["ID", "YEAR", "MONTH", "ELEMENT"]

    data_header_col_specs = [(0, 11), (11, 15), (15, 17), (17, 21)]

    data_header_dtypes = {"ID": str, "YEAR": int, "MONTH": int, "ELEMENT": str}

    data_col_names = [
        [
            "VALUE" + str(i + 1),
            "MFLAG" + str(i + 1),
            "QFLAG" + str(i + 1),
            "SFLAG" + str(i + 1),
        ]
        for i in range(31)
    ]
    data_col_names = sum(data_col_names, [])

    data_replacement_col_names = [
        [
            ("VALUE", i + 1),
            ("MFLAG", i + 1),
            ("QFLAG", i + 1),
            ("SFLAG", i + 1),
        ]
        for i in range(31)
    ]
    data_replacement_col_names = sum(data_replacement_col_names, [])
    data_replacement_col_names = pandas.MultiIndex.from_tuples(
        data_replacement_col_names, names=["VARIABLE", "DAY"]
    )

    data_col_specs = [
        [
            (21 + i * 8, 26 + i * 8),
            (26 + i * 8, 27 + i * 8),
            (27 + i * 8, 28 + i * 8),
            (28 + i * 8, 29 + i * 8),
        ]
        for i in range(31)
    ]
    data_col_specs = sum(data_col_specs, [])

    data_col_dtypes = [
        {
            "VALUE" + str(i + 1): int,
            "MFLAG" + str(i + 1): str,
            "QFLAG" + str(i + 1): str,
            "SFLAG" + str(i + 1): str,
        }
        for i in range(31)
    ]

    # updating dtypes
    data_header_dtypes.update(
        {k: v for d in data_col_dtypes for k, v in d.items()}
    )

    # reading
    dataframe = pandas.read_fwf(
        file,
        colspecs=data_header_col_specs + data_col_specs,
        names=data_header_names + data_col_names,
        index_col=data_header_names,
        dtype=data_header_dtypes,
    )

    if variables is not None:
        if not isinstance(variables, list):
            variables = [variables]
        dataframe = dataframe[
            dataframe.index.get_level_values("ELEMENT").isin(variables)
        ]

    # naming
    dataframe.columns = data_replacement_col_names

    if not flags:
        dataframe = dataframe.loc[:, ("VALUE", slice(None))]

    # stacking
    dataframe = dataframe.stack(level="DAY").unstack(level="ELEMENT")

    # nameing
    names = [i[0] if i[0] != "VALUE" else i[1] for i in dataframe.columns]

    # adjustment
    dataframe.columns = dataframe.columns.droplevel("VARIABLE")
    dataframe.columns = names

    # replace the entire index with the date
    dataframe.index = pandas.to_datetime(
        dataframe.reset_index().loc[:, ["YEAR", "MONTH", "DAY"]],
        errors="coerce",
    )

    # removing invalid dates
    dataframe = dataframe.loc[numpy.invert(pandas.isnull(dataframe.index))]

    # renaming
    dataframe.index.name = "datetime"

    return dataframe
