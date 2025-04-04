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
import re

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def frequency(*args):
    """frequency function

    args -> input(s) to parse frequency.

    """

    # converting to list
    args = list(args)

    for i, arg in enumerate(args):

        if type(arg).__name__ in ["Series", "DataFrame"]:
            args[i] = args[i].index

        if args[i].freq is None:
            args[i] = args[i].to_series().diff().mode()[0]
        else:
            args[i] = args[i].freq

    if len(args) == 1:
        return args[0]
    else:
        return tuple(args)


def decoder(*args):
    """decoder function

    args -> input(s) to decode.

    """

    # converting to list
    args = list(args)

    # initializing lists
    unit = []
    step = []

    for i, arg in enumerate(args):

        # parsing time unit
        unit.append(
            str("".join(i for i in arg if not i.isdigit()))
            if any(not i.isdigit() for i in arg)
            else None
        )
        # parsing time step
        step.append(
            int("".join(i for i in arg if i.isdigit()))
            if any(i.isdigit() for i in arg)
            else None
        )

    if len(args) == 1:
        return unit[0], step[0]
    else:
        return unit, step


def timestring(*args):
    """timestring function

    args -> input(s) to timestring.

    """

    # converting to list
    args = list(args)

    # initializing timedelta
    timedelta = [[]] * len(args)

    # initializing units
    years, months, days, hours, minsec, minutes, seconds = (
        [None] * len(args) for _ in range(7)
    )

    # initializing parts
    parts = [[]] * len(args)

    # initializing output
    output = [None] * len(args)

    for i, arg in enumerate(args):

        # timedelta
        timedelta[i] = pandas.Timedelta(arg)

        # values
        years[i] = timedelta[i].days // 365
        months[i] = timedelta[i].days % 365 // 30
        days[i] = timedelta[i].days % 365 % 30
        hours[i], minsec[i] = divmod(timedelta[i].seconds, 3600)
        minutes[i], seconds[i] = divmod(minsec[i], 60)

        if years[i]:
            parts[i].append(f"{years[i]} years")
        if months[i]:
            parts[i].append(f"{months[i]} months")
        if days[i]:
            parts[i].append(f"{days[i]} days")
        if hours[i]:
            parts[i].append(f"{hours[i]} hours")
        if minutes[i]:
            parts[i].append(f"{minutes[i]} minutes")
        if seconds[i]:
            parts[i].append(f"{seconds[i]} seconds")

        if len(output) != 0:
            output[i] = ", ".join(parts[i])

    if len(args) == 1:
        return output[0]
    else:
        return tuple(output)


def formatter(*args, spaces=None, digits=None, units=None, textcase=None):
    """formatter function

    args     -> input(s) to format;
    spaces   -> spaces lenght;
    digits   -> digits lenght;
    units    -> flagging units formatting;
    textcase -> optional upper/lower case modifier;

    """

    # converting to list
    args = list(args)

    if isinstance(textcase, str):
        textcase = getattr(str, textcase, None)
    else:
        textcase = lambda x: x

    for i, arg in enumerate(args):

        # timestring
        args[i] = timestring(arg)

        if isinstance(spaces, int):
            space = " " * spaces
            args[i] = args[i].replace(" ", space)
        if isinstance(digits, int):
            number = re.search(r"\d+", args[i]).group()
            args[i] = args[i].replace(number, f"{int(number):0{digits}d}")
        if units:
            unit = {
                "years": "y",
                "months": "m",
                "days": "d",
                "hours": "H",
                "minutes": "M",
                "seconds": "S",
            }
            for k, v in unit.items():
                args[i] = args[i].replace(k, textcase(v))

    if len(args) == 1:
        return args[0]
    else:
        return tuple(args)


def converter(*args, output=None):
    """converter function

    args   -> input(s) to convert;
    output -> optional value/unit output selection:

    """

    # formating inputs
    args = formatter(
        *args,
        spaces=None,
        digits=None,
        units=True,
    )

    if output is None:
        output = [
            (int(re.search(r"\d+", arg).group()), arg[-1]) for arg in args
        ]
    elif output == "value":
        output = [int(re.search(r"\d+", arg).group()) for arg in args]
    elif output == "unit":
        output = [arg[-1] for arg in args]

    if len(output) == 1:
        return output[0]
    else:
        return output


def array(
    start,
    stop,
    unit=None,
    step=None,
    timestring=None,
    index=None,
    complete=True,
    fixed=True,
):
    """array function

    start      -> start date/year;
    stop       -> stop date/year;
    unit       -> time delta unit;
    step       -> time delta step;
    timestring -> optional timestring input if no unit/step provided;
    index      -> optional checking index;
    complete   -> flag for complete dates;
    fixed      -> flag for fixed dates.

    """

    if timestring is not None:
        unit, step = decoder(timestring)

    if unit == "y":
        offset = pandas.DateOffset(years=step)
        form = "%Y"
    if unit == "m":
        offset = pandas.DateOffset(months=step)
        form = "%Y-%m"
    if unit == "d":
        offset = pandas.DateOffset(days=step)
        form = "%Y-%m-%d"

    if complete:
        form = "%Y-%m-%d"

    # date formatting
    dates_i = pandas.to_datetime(
        f"{start}-01-01" if isinstance(start, int) else start
    )
    dates_j = pandas.to_datetime(
        f"{stop}-01-01" if isinstance(stop, int) else stop
    )

    if fixed:
        block_i = dates_i.replace(month=1, day=1)
    else:
        block_i = dates_i

    # initialize
    dates = []

    while block_i <= dates_j:
        block_j = block_i + offset - pandas.DateOffset(days=1)
        if index is None or any((block_i <= index) & (index <= block_j)):
            dates.append(
                [block_i.strftime(form), min(block_j, dates_j).strftime(form)]
            )
        block_i = block_j + pandas.DateOffset(days=1)

    if fixed:
        dates[-1][-1] = dates_j.replace(month=12, day=31).strftime(form)

    return numpy.asarray(dates)


def counter(data, unit="y", column="datetime", dataframe=True):
    """counter function

    data      -> input array/series/dataframe;
    column    -> label of data column;
    time      -> counter time unit;
    dataframe -> flagging dataframe output.

    """

    if (
        str(data.__class__) == "<class 'pandas.core.series.Series'>"
        or str(data.__class__) == "<class 'pandas.core.frame.DataFrame'>"
    ):
        if data.index.name == column:
            data.reset_index(inplace=True)

        if column is None:
            data = data.to_numpy()
        else:
            data = data.loc[:, column].to_numpy()

    # dates
    dates = pandas.DatetimeIndex(data)

    if unit == "y":
        value = dates.year.unique().size
        label = "years"
    if unit == "m":
        value = dates.month.unique().size
        label = "months"
    if unit == "d":
        value = dates.day.unique().size
        label = "days"

    if dataframe:
        return pandas.DataFrame(data=[value], columns=[label])
    else:
        return value, label
