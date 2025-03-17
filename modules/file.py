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

import os
import pandas
import pathlib
import pickle

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def path(
    filename=None,
    subdir=None,
    extension=None,
    mode="parent",
    check=True,
    make=True,
):
    """path function"""

    if mode == "absolute":
        pardir = None
    elif mode == "relative":
        pardir = os.path.abspath(os.getcwd())
    elif mode == "parent":
        pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    if isinstance(filename, (list, tuple)):
        filename = utils.concatenating(*filename, separator="_")

    if isinstance(subdir, (list, tuple)):
        subdir = utils.concatenating(*subdir, separator="/")

    # path
    path = utils.concatenating(pardir, subdir, separator="/")

    if path is not None:

        if filename is not None and path in filename and check:
            path = None
        elif make:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # path
    path = utils.concatenating(path, filename, separator="/")

    if (
        extension is not None
        and path is not None
        and "." + extension in path
        and check
    ):
        extension = None

    return utils.concatenating(path, extension, separator=".")


def split(
    filename,
    subdir=None,
    extension=None,
    split="filename",
    mode="parent",
):
    """data splitting"""

    if split == "filename":
        return [
            path(i, subdir, extension, mode)
            for i in listing(subdir, filtering=extension, mode=mode)
            if filename in i
        ]
    elif split == "directory":
        return [
            path(filename, [subdir, i], extension, mode)
            for i in listing(subdir, "directory", mode=mode)
        ]


def flag(filename, subdir="data", extension="pkl", mode="parent"):
    """data flagging"""

    return os.path.isfile(path(filename, subdir, extension, mode))


def remove(filename, subdir="data", extension="pkl", mode="parent"):
    """ "data removing"""

    # removing
    os.remove(path(filename, subdir, extension, mode))


def listing(directory, output=None, filtering=None, split=None, mode="parent"):
    """data listing"""

    # items
    items = os.listdir(path(subdir=directory, mode=mode))

    if output == "directory":
        items = [
            item
            for item in items
            if os.path.isdir(path(subdir="directory/" + item, mode=mode))
        ]
    else:
        items = [
            item
            for item in items
            if os.path.isfile(path(item, subdir=directory, mode=mode))
        ]

        if output == "filename":
            items = [os.path.splitext(item)[0] for item in items]
        elif output == "extension":
            items = [os.path.splitext(item)[1][1:] for item in items]

    if isinstance(filtering, str):
        items = [item for item in items if filtering in item]

    if isinstance(split, str):
        items = [item.split(split) for item in items]

    return items


def opener(compression=None):
    """data opener"""

    if compression is None:
        return open
    elif compression == "gzip":
        import gzip

        return gzip.open
    elif compression == "lzma":
        import lzma

        return lzma.open


def load(
    filename,
    subdir=None,
    extension="pkl",
    compression=None,
    mode="parent",
):
    """data loading"""

    if type(filename).__name__ == "BufferedReader":
        return pickle.load(filename)
    else:
        with opener(compression)(
            path(filename, subdir, extension, mode), "rb"
        ) as dump:
            return pickle.load(dump)


def save(
    data,
    filename,
    subdir=None,
    extension="pkl",
    compression=None,
    mode="parent",
):
    """data saving"""

    if type(filename).__name__ == "BufferedWriter":
        pickle.dump(filename)
    else:
        with opener(compression)(
            path(filename, subdir, extension, mode), "wb"
        ) as dump:
            pickle.dump(data, dump)


def read(
    filename,
    subdir=None,
    extension="csv",
    compression="infer",
    mode="parent",
    separator=",",
    header="infer",
    index=None,
    columns=None,
    skipspaces=True,
    skipheader=0,
    skipfooter=0,
    dtype=None,
    chunksize=None,
    **keywords,
):
    """read function"""

    # reading dataframe
    dataframe = pandas.read_csv(
        path(filename, subdir, extension, mode),
        compression=compression,
        sep=separator,
        header=header,
        index_col=index,
        usecols=columns,
        skipinitialspace=skipspaces,
        skiprows=skipheader,
        skipfooter=skipfooter,
        dtype=dtype,
        chunksize=chunksize,
        **keywords,
    )

    if chunksize is None:
        return dataframe
    else:
        return pandas.concat(dataframe)


def export(
    dataframe,
    filename,
    subdir=None,
    extension="csv",
    compression=None,
    mode="parent",
    separator=",",
    columns=None,
    index=True,
    formatfloat="%.6f",
    formatdate=None,
    **keywords,
):
    """export function"""

    # export dataframe
    dataframe.to_csv(
        path(filename, subdir, extension, mode),
        compression=compression,
        sep=separator,
        columns=columns,
        index=index,
        float_format=formatfloat,
        date_format=formatdate,
        **keywords,
    )
