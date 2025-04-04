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
import importlib
import itertools
import multiprocessing
import numpy
import os
import pandas

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import config
import file

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def load(
    filename,
    subdir="data",
    extension="pkl",
    compression=None,
    path="parent",
    module=True,
    existence=True,
    level=0,
    **keywords,
):
    """load function

    filename    -> data filename;
    subdir      -> optional subdir;
    extension   -> file extension;
    compression -> file compression;
    path        -> mode for path definition;
    module      -> loading module flag/name for infer/explicit definition;
    existence   -> flagging existence check;
    level       -> log level;
    keywords    -> function keywords.

    """

    if extension is True and module is True:
        extension = os.path.splitext(filename)[1][1:]

    if not existence or file.flag(filename, subdir, extension, path):

        try:
            if extension == "pkl" or module == "pickle":
                out = file.load(
                    filename, subdir, extension, compression, path, **keywords
                )
            elif extension == "csv" or module == "pandas":
                out = file.read(
                    filename, subdir, extension, compression, path, **keywords
                )
            else:
                # file path
                path = file.path(filename, subdir, extension, path)

                if extension == "shp" or module == "geopandas":
                    import geopandas

                    out = geopandas.read_file(path, **keywords)
                elif extension == "nc" or module == "xarray":
                    import xarray

                    out = xarray.open_dataset(path, **keywords)
                else:
                    if isinstance(module, str):
                        module = definition(module)
                    elif isinstance(module, (list, tuple)):
                        module = definition(*module)
                    elif isinstance(module, dict):
                        module = definition(**module)

                    out = module(path, **keywords)

        except:
            printing(
                "load: processing",
                filename,
                "failed",
                level=level,
            )
        else:
            printing(
                "load: processing",
                filename,
                "success",
                level=level,
            )
            return out
    else:
        printing("load: file", filename, "nonexistent", level=level)


def save(
    data,
    filename,
    subdir="data",
    extension="pkl",
    compression=None,
    path="parent",
    module=True,
    level=0,
    **keywords,
):
    """save function

    data        -> data object;
    filename    -> data filename;
    subdir      -> optional subdir;
    extension   -> file extension;
    compression -> file compression;
    path        -> mode for path definition;
    module      -> saving module flag/name for infer/explicit definition;
    level       -> log level;
    keywords    -> function keywords.

    """

    if extension is True:
        extension = os.path.splitext(filename)[1][1:]

    try:
        if extension == "pkl":
            file.save(
                data,
                filename,
                subdir,
                extension,
                compression,
                path,
                **keywords,
            )
        elif extension == "csv":
            file.export(
                data,
                filename,
                subdir,
                extension,
                compression,
                path,
                **keywords,
            )
        else:
            # file path
            path = file.path(filename, subdir, extension, path)
            if extension == "shp":
                data.to_file(path, **keywords)
            elif extension == "nc":
                data.to_netcdf(path, **keywords)
            else:
                if isinstance(module, str):
                    module = definition(module)
                elif isinstance(module, (list, tuple)):
                    module = definition(*module)
                elif isinstance(module, dict):
                    module = definition(**module)

                    module(path, **keywords)

    except:
        printing("save: processing", filename, "failed", level=level)
    else:
        printing("save: processing", filename, "success", level=level)


def concatenating(*args, discard=None, separator=" "):
    """concatenating function

    args      -> input sequence;
    discard   -> discarded elements;
    separator -> concatenation separator.

    """

    if not isinstance(args, list):
        args = list(args)
    if not isinstance(separator, str):
        separator = str(separator)

    if not isinstance(discard, (list, tuple)):
        discard = [discard]

    # discarding elements
    args = [i for i in args if i not in discard]

    for i in range(len(args)):
        if not isinstance(args[i], list):
            args[i] = [args[i]]

    # extract
    args = [j for i in args for j in i]

    # string
    string = separator.join([str(i) for i in args])

    if string == "":
        return None
    else:
        return string


def configuring(
    name,
    subdir="configuration",
    extension="txt",
    compression=None,
    path="parent",
    label=None,
    typology=None,
):
    """configuring function

    name        -> name of the configuration;
    subdir      -> configuration subdir;
    extension   -> configuration extension;
    compression -> configuration compression;
    path        -> mode for path definition;
    label       -> entry(es) extracting label;
    typology    -> entry(es) formatting typology.

    """

    # list
    out = config.listing(name, subdir, extension, compression, path)

    if label is not None:
        out = config.extracting(out, label)

    if typology is not None:
        out = config.formatting(out, typology)

    if len(out) == 1:
        return out[0]
    else:
        return out


def dictionary(
    name,
    *args,
    update=None,
    subdir="dictionaries",
    extension="py",
    compression=None,
    path="parent",
    dictionary="dictionary",
    level=0,
):
    """dictionary function

    name        -> name of the dictionary;
    args        -> arguments of the dictionary;
    update      -> extra keywords;
    subdir      -> dictionary subdir;
    extension   -> dictionary extension;
    compression -> configuration compression;
    path        -> mode for path definition;
    dictionary  -> variable label;
    level       -> log level;

    """

    if file.flag(name, subdir, extension, path):

        # initializing contest
        context = {}

        try:

            # dictionary execution
            exec(
                file.opener(compression)(
                    file.path(name, subdir, extension, path)
                ).read(),
                context,
            )

            if dictionary not in context:
                printing("dictionary: variable", name, "invalid", level=level)
            else:
                dictionary = context[dictionary]

            if len(args) > 0:
                for i in args:
                    dictionary = dictionary[i]

            if update is not None:
                dictionary.update(update)

        except:
            printing(
                "dictionary: importing", name, *args, "failed", level=level
            )
        else:
            printing(
                "dictionary: importing", name, *args, "success", level=level
            )

        return dictionary

    else:
        printing("dictionary: file", name, *args, "nonexistent", level=level)


def scriptkeys(*args, update=None, separator=":", split=","):
    """scriptkeys function

    args      -> input arguments;
    update    -> optional globals/locals/dictionary to update;
    separator -> key:value:type:subtype separator;
    split     -> list/tuple/iterator values split character.

    """

    # parsing keys
    keys = {
        k: (
            v
            if len(types) == 0
            else (
                eval(types[0])(eval(types[1])(i) for i in v.split(split))
                if types[0] in ["list", "tuple", "iter"]
                else eval(types[0])(v)
            )
        )
        for arg in args
        if separator in arg
        for k, v, *types in [arg.split(separator)]
    }

    if isinstance(update, dict):
        update.update(keys)
    else:
        return keys


def printing(*args, level=1, verbousity=0):
    """printing function

    args       -> function arguments;
    level      -> logging level;
           = 0 -> disabled;
           = 1 -> brief;
           = 2 -> normal;
           = 3 -> detailed;
    verbousity -> logging verbousity.

    """

    if level <= verbousity:
        print(*args)


def aligner(*args):
    """aligner function

    args -> arguments to be dimensionally aligned lists.

    """

    # converting to lists
    args = [i if isinstance(i, list) else [i] for i in args]

    # max number of inputs
    n = max(len(i) for i in args)

    return tuple([i if len(i) == n else i * n for i in args])


def reshaper(x, *X, n=1, s=2):
    """array(s) reshape function

    x -> array to be reshaped;
    X -> optional array(s) to be dimensionally coherent;
    n -> number of variables;
    s -> dimensional size.

    """

    if s < 2:
        return x, *X

    if isinstance(x, (int, float)):
        x = numpy.array([x])
    elif type(x).__name__ != "ndarray":
        x = numpy.array(x)

    if x.ndim <= 1:
        x = x.reshape((1, n, *tuple(i for i in [1] * (s - 2))))
    elif x.ndim == 2:
        x = x.reshape((-1, n, *tuple(i for i in [1] * (s - 2))))

    if len(X) == 0:
        return x
    else:
        return x, *[
            (
                x.reshape(-1, 1, *tuple(i for i in [1] * (s - 2)))
                if type(x).__name__ == "ndarray"
                else x
            )
            for x in X
        ]


def ranges(pivot, resolution=1, add=None, unique=True, sort=None, dtype=float):
    """ranges function

    pivot      -> pivot values for ranges;
    resolution -> resolution of each range;
    add        -> optional values to be added to list;
    unique     -> flagging uniqueness;
    sort       -> sorting flag;
    dtype      -> data type.

    """

    # number of ranges
    n = len(pivot) - 1

    if isinstance(resolution, (int, float)):
        resolution = [resolution] * n
    elif not isinstance(resolution, list):
        resolution = list(resolution)

    # ranges array
    ranges = numpy.arange(pivot[0], pivot[1], resolution[0])

    if n > 1:
        ranges = numpy.concatenate(
            [
                ranges,
                *(
                    numpy.arange(pivot[i], pivot[i + 1], resolution[i])
                    for i in range(1, n)
                ),
            ],
            axis=None,
        )

    if add is not None:
        ranges = numpy.append(ranges, add)
    if unique:
        ranges = numpy.unique(ranges)
    if sort:
        ranges = numpy.sort(ranges)
    if dtype:
        ranges = ranges.astype(dtype)

    return ranges


def indexer(*args, dtype=None, intersection=None):
    """indexer function

    args         -> input arguments;
    dtype        -> optional index data type;
    intersection -> flagging itnersection output.

    """

    # converting to list
    args = list(args)

    if not isinstance(dtype, list):
        dtype = [dtype] * len(args)

    for i in range(len(args)):
        if type(args[i]).__name__ == "ndarray":
            args[i] = args[i].reshape(-1)
        elif (
            not isinstance(args[i], (list, tuple, range))
            and type(args[i]).__name__ != "Series"
            and type(args[i]).__name__ != "Index"
        ):
            args[i] = [args[i]]
        if type(args[i]).__name__ != "Index":
            args[i] = pandas.Index(args[i])
        if dtype[i] is not None:
            args[i] = args[i].astype(dtype[i])

    if intersection:
        return functools.reduce(lambda i, j: i.intersection(j), args)
    elif len(args) == 1:
        return args[0]
    else:
        return tuple(args)


def iterables(*args, packed=None, product=None, index=None):
    """iterables function

    args    -> iterable arguments;
    packed  -> flagging forced packed output;
    product -> flagging iterable from productory;
    index   -> index return flag/constant.

    """

    if packed is None and len(args) == 1:
        iterables = tuple(args[0])
    else:
        if product is True:
            iterables = itertools.product(*args)
        else:
            iterables = zip(*args)

    if index is None:
        return iterables
    else:
        k = 0 if index is True else (index if isinstance(index, int) else None)
        return (
            (i + k, *j) if isinstance(j, tuple) else (i + k, j)
            for i, j in enumerate(iterables)
        )


def filtering(flags, *args):
    """filtering function

    flags -> flags iterable;
    args  -> filtered arguments.

    """

    # converting to list
    args = list(args)

    for i, arg in enumerate(args):
        if isinstance(arg, list):
            args[i] = [arg[i] for i, j in enumerate(flags) if j]
        else:
            args[i] = arg[flags]

    if len(args) == 1:
        return args[0]
    else:
        return tuple(args)


def enumerating(string, separator=None):
    """enumerating function

    string    -> string to be enumerated;
    separator -> string/number separator.

    """

    # defining prefix
    prefix = "".join(filter(str.isalpha, string))

    # defining number
    number = int("".join(filter(str.isdigit, string)))

    if separator is not None:
        prefix = prefix + separator

    return [prefix + str(i) for i in range(1, number + 1)]


def reducing(*args, operation=None):
    """reducing function

    args      -> input arguments;
    operation -> reducing function string.

    """

    return functools.reduce(lambda i, j: getattr(i, operation)(j), args)


def iterfun(function, *args, dictionaries={}, **keywords):
    """iterfun function

    function     -> function definition;
    args         -> global arguments;
    dictionaries -> set of dictionaries;
    keywords     -> global keywords.

    """

    if not isinstance(dictionaries, list):
        dictionaries = [dictionaries]

    # initialize
    results = []

    for dictionary in dictionaries:
        # udate dictionary
        dictionary.update(keywords)

        # append output
        results.append(function(*args, **dictionary))

    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def process(function, *args, product=True, drop=True, cpu=1, **keywords):
    """process function

    function  -> process function;
    args      -> process arguments;
    product   -> flagging iterable from productory;
    drop      -> drop invalid outputs;
    cpu       -> number of cpus.

    """

    # iterables items
    items = iterables(*args, packed=True, product=product)

    if cpu == 1:
        results = [function(*item) for item in items]
    else:
        with multiprocessing.Pool(processes=cpu) as pool:
            results = pool.starmap(function, items)

    if drop:
        results = [result for result in results if result is not None]

    if len(results) == 0:
        return
    else:
        return results


def timeout(seconds=10):
    """timeout function

    seconds -> timeout seconds.

    """

    import multiprocessing.pool

    def decorator(item):
        """wrap the original function"""

        @functools.wraps(item)
        def wrapper(*args, **kwargs):
            """closure for function"""

            thread = multiprocessing.pool.ThreadPool(processes=1)
            results = thread.apply_async(item, args, kwargs)

            return results.get(seconds)

        return wrapper

    return decorator


def definition(
    module=None, function=None, subdir="modules", extension="py", path="parent"
):
    """definition function

    module    -> name of the module;
    function  -> name of the function;
    subdir    -> module subdir;
    extension -> module extension;
    path      -> mode for path definition.

    """

    if isinstance(module, (int, str)):

        try:
            # module import
            module = importlib.import_module(module)
        except:
            # module spec
            spec = importlib.util.spec_from_file_location(
                module, file.path(module, subdir, extension, path)
            )

            # module object
            module = importlib.util.module_from_spec(spec)

            # execution
            spec.loader.exec_module(module)

    if function is None:
        return module
    else:
        if isinstance(function, str):
            function = [function]
        # getting output
        out = module
        for fun in function:
            out = getattr(out, fun, None)
        if out is None:
            raise
        else:
            return out
