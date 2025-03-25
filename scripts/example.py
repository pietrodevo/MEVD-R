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

import sys

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MODULES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

sys.path.append("../modules")

import region
import utils

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""EXECUTABLE"""

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

"""SETTINGS"""

# optional codenames
datacode = None
extrcode = None
cluscode = None
valicode = None
plotcode = None

# country name
country = "us"

# reader name
reader = "ghcnd"

# processing skipping
skip = None

# cpu used
cpu = 1

# analyses codes:
analyses = ["events", "clustering", "cross", "plot", "supercross", "superplot"]
# analyses = ["events"]
# analyses = ["clustering"]
# analyses = ["cross"]
analyses = ["plot"]
# analyses = []

# processing mode
procset = None

# processing resolution
resolution = "1h"

# selected durations
if resolution == "1h":
    # durations = ["01h", "02h", "03h", "06h", "09h", "12h", "24h"]
    durations = ["01h"]
if resolution == "1d":
    # durations = ["01d", "02d", "03d"]
    durations = ["01d"]

# minimum years
if resolution == "1h":
    # years = 50
    years = 60
    # years = 70
if resolution == "1d":
    # years = 100
    years = 120
    # years = 140

# data threshold
threshold = None

# clustering distribution
clustering = "weibull"

# output process
process = "rmse"
process = "high"
# process = None

# output compression
compression = "gzip"
# compression = None

# output extension
# extension = "csv"
extension = "pkl"

# validation benchmarks
# benchmarks = ["site", "envelope", "region"]
benchmarks = ["site", "envelope"]
# benchmarks = ["envelope"]

# validation sizes
sizes = [3, 4, 5]
sizes = 5

# validation portions
portions = [3, 5, 7]
portions = 3

# validations distributions
distributions = ["gev","mev"]

# validation samples
samples = 1000

# validation posts
posts = {} if process is None else utils.dictionary("posts", process)

# metrics chunks
chunksize = 50000

# metrics types
dtype = utils.dictionary("dtypes", "metrics")

# path definitions
datadir = utils.concatenating("data", country, datacode, separator="/")
extrdir = utils.concatenating("extraction", country, extrcode, separator="/")
clusdir = utils.concatenating("clusters", country, cluscode, separator="/")
validir = utils.concatenating("validation", country, valicode, separator="/")
plotdir = utils.concatenating("plot", country, plotcode, separator="/")

"""KEYS"""

utils.scriptkeys(*sys.argv, update=globals(), separator=":", split=",")

"""ANALYSES"""

"""database"""

# stations dataframe
dtb = utils.load(country.upper() + "_" + "stations", "meta", "csv", separator=";", index="code", dtype={"code": str})

# stations index
stations = dtb.index.unique()

# breakpoint()

"""events"""

if "events" in analyses:

    # extraction
    dic_series = utils.dictionary("series", country, reader)
    region.stormwalker(dtb, stations=None, events=None, durations=durations, **dic_series, skip=None, procdir=[extrdir, durations[0]], proczip=compression, cpu=cpu)

    # data years
    lengths = region.datafetcher(dtb, stations, drop="any", filename=["lengths", durations[0]], outdir=datadir, subdir=[extrdir, durations[0]], subzip=compression, cpu=cpu, function=["dates", "counter"])

    # breakpoint()

"""filtering"""

# station lengths
lengths = utils.load(["lengths", durations[0]], datadir, "csv", index="code", dtype={"code": str})

# filtered stations
stations = lengths.loc[lengths.loc[:, "years"] >= years, :].index

# breakpoint()

# %%

"""clustering"""

if "clustering" in analyses:

    # envelope
    dic_clusters = utils.dictionary("clusters", country, "weibull", update={"infer": True})
    clusters = region.clusters(dtb, [extrdir, durations[0]], stations, durations=durations, **dic_clusters, skip=skip, out=True, filename=["clusters", durations[0], "infer"], outdir=datadir, subzip=compression, procdir=[clusdir, durations[0], "infer"], proczip=compression, procset="difference", cpu=cpu)

    # region
    dic_clusters = utils.dictionary("clusters", country, "weibull", update={"infer": False})
    clusters = region.clusters(dtb, [extrdir, durations[0]], stations, durations=durations, **dic_clusters, skip=skip, out=True, filename=["clusters", durations[0]], outdir=datadir, subzip=compression, procdir=[clusdir, durations[0]], proczip=compression, procset="difference", cpu=cpu)

    # breakpoint()

# %%

"""crossvalidation"""

if "cross" in analyses:

    if "site" in benchmarks:
        stations_site = lengths.loc[lengths.loc[:, "years"] >= years].index.unique()
        region.jocker(dtb, [extrdir, durations[0]], stations=stations_site, durations=durations, benchmarks="site", lengths=portions, distributions=[i for i in distributions if i != "cmev"] if "mev" in distributions else distributions, threshold=threshold, samples=samples, skip=skip, out=None, filename=None, subzip=compression, procdir=[validir, durations[0]], proczip=compression, procset=procset, cpu=cpu)

    if "envelope" in benchmarks:
        clusters_envelope = utils.load(["clusters", durations[0], "infer"], datadir, "csv", index="code", dtype={"code": str})
        stations_envelope = clusters_envelope.index.unique().intersection(lengths.loc[lengths.loc[:, "years"] >= years].index.unique())
        region.jocker(dtb, [extrdir, durations[0]], stations=stations_envelope, durations=durations, benchmarks="envelope", sizes=sizes, lengths=portions, distributions=distributions, clusters=clusters_envelope, threshold=threshold, samples=samples, distances=True, skip=skip, out=None, filename=None, subzip=compression, procdir=[validir, durations[0]], proczip=compression, procset=procset, cpu=cpu)

    if "region" in benchmarks:
        clusters_region = utils.load(["clusters", durations[0]], datadir, "csv", index="code", dtype={"code": str})
        stations_region = clusters_region.index.unique().intersection(lengths.loc[lengths.loc[:, "years"] >= years].index.unique())
        region.jocker(dtb, [extrdir, durations[0]], stations=stations_region, durations=durations, benchmarks="region", sizes=sizes, lengths=portions, distributions=distributions, clusters=clusters_region, threshold=threshold, samples=samples, distances=True, skip=skip, out=None, filename=None, subzip=compression, procdir=[validir, durations[0]], proczip=compression, procset=procset, cpu=cpu)

    # stations
    stations = utils.indexer(*[globals()["stations_%s" % (i)] for i in benchmarks], intersection=True)

    if process is not None:
        for duration in durations:
            region.jocker(dtb, [extrdir, duration], stations=stations, durations=duration, benchmarks=benchmarks, **posts, skip=True, out=True, filename=["metrics", duration, process], outdir=datadir, procdir=[validir, durations[0]], subzip=compression, proczip=compression, extension=extension, compression=compression, procset="intersection", cpu=cpu)

    # breakpoint()