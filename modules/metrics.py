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


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def divide(data):
    """divide function"""

    # array conversion
    data = numpy.asarray(data)

    return numpy.divide.reduce(data, axis=1)


def subtract(data):
    """subtract function"""

    # array conversion
    data = numpy.asarray(data)

    return numpy.subtract.reduce(data, axis=1)


def absolute(data):
    """absolute function"""

    # array conversion
    data = numpy.asarray(data)

    return numpy.abs(subtract(data))


def square(data):
    """square function"""

    # array conversion
    data = numpy.asarray(data)

    return subtract(data) ** 2


def re(data):
    """relative error function"""

    # array conversion
    data = numpy.asarray(data)

    return subtract(data) / data[:, 1]


def lre(data):
    """logarithm relative error function"""

    # array conversion
    data = numpy.asarray(data)

    return numpy.log10(1 + re(data))


def fse(data):
    """fractional standard function"""

    # array conversion
    data = numpy.asarray(data)

    return numpy.sqrt(numpy.sum(numpy.abs(re(data)) ** 2) / data.shape[0])


def mse(data):
    """mean square error function"""

    # array conversion
    data = numpy.asarray(data)

    return numpy.add.reduce(square(data), axis=0) / data.shape[0]


def rmse(data):
    """root mean square error function"""

    # array conversion
    data = numpy.asarray(data)

    return numpy.sqrt(mse(data))


def skill(data):
    """skill score function"""

    # array conversion
    data = numpy.asarray(data)

    return (
        numpy.corrcoef(data[:, 0], data[:, 1])[0, 1] ** 2
        - (
            numpy.corrcoef(data[:, 0], data[:, 1])[0, 1]
            - (numpy.std(data[:, 0]) / numpy.std(data[:, 1]))
        )
        ** 2
        - (
            numpy.subtract.reduce(numpy.mean(data, axis=0), axis=0)
            / numpy.std(data[:, 1])
        )
        ** 2
    )
