# MEVD-R: Metastatistical Extreme Value Distribution Regional Framework  

[![DOI](https://img.shields.io/badge/DOI-10.22541%2Fessoar.173193565.59169660%2Fv1-blue)](https://doi.org/10.22541/essoar.173193565.59169660/v1)  

> **Estimates of rare rainfall extremes in ungauged areas**  
> Pietro Devò, Maria Francesca Caruso, Marco Borga, Marco Marani  

## Description  

This repository contains the code and structure of the **MEVD-R (Metastatistical Extreme Value Distribution Regional)** framework, developed for estimating extreme rainfall events in ungauged areas.  

Documentation is incomplete and in progress.

## Structure  

The framework follows this directory structure:

```plaintext
📂 MEVD-R/
├── 📂 clusters/      # Cluster analysis outputs
├── 📂 data/          # Output data storage
├── 📂 dictionaries/  # Python dictionaries with parameters and settings
├── 📂 extraction/    # Extracted independent events from raw time series
├── 📂 meta/          # Database of stations and main reference information
├── 📂 modules/       # Python modules specific to the framework
├── 📂 scripts/       # Main analysis scripts
├── 📂 series/        # Raw time series data from various datasets
└── 📂 validation/    # Cross-validation outputs
```

## Dependencies  

The framework requires the following Python libraries:  

- [`numpy`](https://numpy.org/)    
- [`pandas`](https://pandas.pydata.org/)    
- [`scipy`](https://scipy.org/)    

## Contact  

For any inquiries, please contact the author:  
📧 E-MAIL: [pietro.devo@dicea.unipd.it](mailto:pietro.devo@dicea.unipd.it)  
🔗 ORCID: https://orcid.org/0009-0005-7860-9910

## License & Citation  

This code is open-source and can be used, modified, and distributed freely. If you use this framework in your work, please cite the associated paper.  

📌 **Recommended citation:**  
Devò, P., Caruso, M. F., Borga, M., & Marani, M. (2024). *Estimates of rare rainfall extremes in ungauged areas*.  
DOI: [10.22541/essoar.173193565.59169660/v1](https://doi.org/10.22541/essoar.173193565.59169660/v1).  
