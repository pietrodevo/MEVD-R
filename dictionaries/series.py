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
"""DICTIONARY"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

# initialize
dictionary = {}

# australia
dictionary["as"] = {}
dictionary["as"]["ghcnd"] = {
    "subdir": "series/as/ghcnd/01d",
    "subext": "dly",
    "variables": "PRCP",
    "reader": "ghcnd",
    "separator": None,
    "decimal": ".",
    "flags": {"QFLAG": None},
    "nans": -9999,
    "readerdic": {"frequency": "01d"},
    "rename": {"PRCP": "value"},
    "scale": {"value": 0.1},
    "frequency": "01d",
    "percentile": 0.5,
    "length": 20,
    "lag": "09h",
    "threshold": 0.2,
}

# europe
dictionary["eu"] = {}
dictionary["eu"]["eca"] = {
    "subdir": "series/eu/eca/01d",
    "subext": "txt",
    "skipheader": 20,
    "skipfooter": 0,
    "variables": "RR",
    "reader": None,
    "separator": None,
    "dateformat": "%Y%m%d",
    "decimal": ".",
    "flags": {"Q_RR": 0},
    "nans": -9999,
    "rename": {"DATE": "datetime", "RR": "value"},
    "scale": {"value": 0.1},
    "frequency": "01d",
    "percentile": 0.5,
    "length": 20,
    "lag": "09h",
    "threshold": 0.2,
}

# united states
dictionary["us"] = {}
dictionary["us"]["ghcnd"] = {
    "subdir": "series/us/ghcnd/01d",
    "subext": "dly",
    "variables": "PRCP",
    "reader": "ghcnd",
    "separator": None,
    "decimal": ".",
    "flags": {"QFLAG": None},
    "nans": -9999,
    "readerdic": {"frequency": "01d"},
    "rename": {"PRCP": "value"},
    "scale": {"value": 0.1},
    "frequency": "01d",
    "percentile": 0.5,
    "length": 20,
    "lag": "09h",
    "threshold": 0.2,
}
