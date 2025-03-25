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
dictionary["as"]["gev"] = {
    "realizations": 500,
    "heterogeneity": 0.5,
    "infer": None,
    "years": None,
    "cumulative": 100,
    "dimension": 5,
    "radius": 100,
    "maxima": "y",
    "distribution": "gev",
}
dictionary["as"]["weibull"] = {
    "realizations": 500,
    "heterogeneity": 0.5,
    "infer": None,
    "years": None,
    "cumulative": 100,
    "dimension": 5,
    "radius": 100,
    "maxima": None,
    "distribution": "weibull",
}

# europe
dictionary["eu"] = {}
dictionary["eu"]["gev"] = {
    "realizations": 500,
    "heterogeneity": 0.5,
    "infer": None,
    "years": None,
    "cumulative": 100,
    "dimension": 5,
    "radius": 100,
    "maxima": "y",
    "distribution": "gev",
}
dictionary["eu"]["weibull"] = {
    "realizations": 500,
    "heterogeneity": 0.5,
    "infer": None,
    "years": None,
    "cumulative": 100,
    "dimension": 5,
    "radius": 100,
    "maxima": None,
    "distribution": "weibull",
}

# united states
dictionary["us"] = {}
dictionary["us"]["gev"] = {
    "realizations": 500,
    "heterogeneity": 0.5,
    "infer": None,
    "years": None,
    "cumulative": 100,
    "dimension": 5,
    "radius": 200,
    "maxima": "y",
    "distribution": "gev",
}
dictionary["us"]["weibull"] = {
    "realizations": 500,
    "heterogeneity": 0.5,
    "infer": None,
    "years": None,
    "cumulative": 100,
    "dimension": 5,
    "radius": 200,
    "maxima": None,
    "distribution": "weibull",
}