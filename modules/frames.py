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

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def reduce(dataframe, values="values", level="level", group="group"):
    """reduce function"""

    # getting keys
    keys = dataframe.reset_index().loc[:, level].unique()

    # preparing dataframe
    dataframe = dataframe.sort_values(level).reset_index().reset_index()

    # fetching column names
    columns = [i for i in dataframe.columns if i != values]

    # unstacking dataframe
    dataframe = dataframe.set_index(columns).unstack(level)

    # dropping index level
    dataframe.columns = dataframe.columns.droplevel(0)

    # reordering keys
    dataframe = dataframe.reindex(columns=keys)

    # adjusting column names
    columns = [i for i in columns if i not in level]

    return (
        dataframe.reset_index()
        .set_index(group)
        .groupby(group)
        .first()
        .reset_index()
        .set_index(columns)
        .droplevel(0)
    )


def expand(dataframe, values="values", keys="level", label="label"):
    """expand function"""

    if isinstance(keys, (int, str)):
        keys = [keys]
    else:
        keys = list(keys)

    if keys != list(dataframe.columns):
        # resetting index
        dataframe.reset_index(inplace=True)

        # setting index
        dataframe.set_index(
            [i for i in dataframe.columns if i not in keys], inplace=True
        )

    if "index" in dataframe.index.names:
        dataframe = dataframe.droplevel("index")

    return (
        dataframe.stack()
        .reset_index()
        .rename(
            columns={0: values, "level_%s" % (dataframe.index.nlevels): label}
        )
    )


def focus(
    dataframe,
    group=None,
    sort=None,
    nth=0,
):
    """focus function"""

    if group is None:
        group = []
    elif isinstance(group, (int, str)):
        group = [group]
    elif not isinstance(group, (list)):
        group = list(group)

    if sort is not None:
        dataframe.sort_values(sort, inplace=True)

    return dataframe.groupby(list(dataframe.index.names) + group).nth(nth)


def aggregate(
    dataframe,
    function=None,
    group=None,
    target=None,
    keep="first",
    arguments=(),
    keywords={},
):
    """aggregate function"""

    if function is not None:
        if isinstance(function, (list, tuple)):
            function = utils.definition(*function)
        elif isinstance(function, dict):
            function = utils.definition(**function)

    if group is None:
        group = list(dataframe.index.names)
    elif not isinstance(group, (list, tuple)):
        group = [group]

    if target is None:
        target = list(dataframe.columns)
    if not isinstance(target, (list, tuple)):
        target = [target]

    if type(dataframe).__name__ == "Series":
        dataframe = dataframe.to_frame()
        dataframe.index.name = "index"

    if not dataframe.index.name is None and not dataframe.index.name == "":
        # getting the index
        index = dataframe.index.names

        # resetting the index
        dataframe.reset_index(inplace=True)

    # labels of group(s)
    group = [i for i in group if i in dataframe.columns]

    # labels of column(s)
    column = [i for i in dataframe.columns if i not in group]

    # aggregation dictionary
    dictionary = {i: keep if i not in target else function for i in column}

    # grouping applying
    dataframe = dataframe.groupby(group, as_index=False).agg(
        dictionary, *arguments, **keywords
    )

    if "index" in locals():
        return dataframe.set_index(index)
    else:
        return dataframe


def apply(
    dataframe,
    function=None,
    columns=None,
    group=None,
    keep=None,
    drop=None,
    label=None,
    sort=None,
    arguments=(),
    keywords={},
):
    """apply function"""

    if isinstance(function, (list, tuple)):
        function = utils.definition(*function)
    elif isinstance(function, dict):
        function = utils.definition(**function)

    if label is None:
        label = function.__name__

    # defining function
    function = functools.partial(function, *arguments, **keywords)

    if group is None:
        # applying function
        dataframe.loc[:, label] = function(
            dataframe.loc[:, columns].to_numpy()
        )

        if keep is True:
            return dataframe
        elif isinstance(keep, (int, str)):
            label = [label, keep]
        elif isinstance(keep, (list, tuple)):
            label = [label] + list(keep)

        return dataframe.loc[:, label]

    else:
        if not isinstance(group, list):
            group = [group]

        if drop:
            dataframe.dropna(subset=columns, inplace=True)

        # getting index
        index = list(dataframe.index.names)

        # coherency
        group = [i for i in group if i not in index and i not in columns]

        return (
            dataframe.reset_index()
            .loc[:, index + group + columns]
            .set_index(index + group)
            .groupby(index + group, sort=sort)
            .apply(function)
            .rename(label)
        )
