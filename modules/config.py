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

import file

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def listing(
    filename,
    subdir="configuration",
    extension="txt",
    compression=None,
    path="parent",
):
    """ "config listing"""

    return (
        file.opener(compression)(file.path(filename, subdir, extension, path))
        .read()
        .split("\n")[:-1]
    )


def extracting(data, label):
    """config extracting"""

    # initialize
    out = []

    for line in data:
        if label in line:
            # splitting
            split = list(line.split(" "))

            # indexing
            index = split.index(label)

            # output
            out.append(split[index + 1])

    return out


def formatting(data, typology):
    """config formatting"""

    if isinstance(data, list):
        return [typology(i) for i in data]

    else:
        return typology(data)
