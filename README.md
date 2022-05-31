# Predicting Housing Prices in Santiago, Chile

Gonzalo Oyanedel Vial (gov@chicagobooth.edu)

Results: https://goyanedelv.github.io/santiago/

## Repo structure

This repo contains the code and data used to run the data science pipeline of the project and the scripts used in the data engineering pipeline (excluding the raw data). For the web-scrapping portion, please review the following public repos: [toctoc-scrapper](https://github.com/goyanedelv/toctoc-scrapper) and [portal-inmobiliario-scrapper](https://github.com/goyanedelv/portal-inmobiliario-scrapper).

### Code (data science pipeline)

- `main.py` (82 lines) contains the main prediction program which can be configured through `config.yaml` and run via command line as:

```shell
python main.py
```

- `config.yaml` (55 lines) contains all the configuration information of the data science pipeline, ranging from what models should be ran, experiment tags, hyperparameter turning and hyperparameter for the extrapolation script.

- `extrapolation.py` (76 lines) this script trains a model using all pricing data and creates a prediction for every block in Santiago.

- `models.py` (191 lines) an auxiliary module with all the pipeline for each of the seven machine learning models.

- `evaluation.py` (94 lines) an auxiliary module with a standardized evaluation function.

- `visualization.R` (41 lines) the only script in R, to create `html` maps of the extrapolated results.

### Data

- `/data/master_table.csv` the resulting data from the data engineering pipeline, ready to be consumed by the machine learning models.

- `/data/predicted_and_actuals_full_df.xlsx` the output of the extrapolation and input for data visualization.

- `/data/arbitrage_analysis.xlsx` additional files regarding the arbitrage analysis.

- `/output/*` all files resulting from hyperparameter tuning experiments.

### Others

- `/data_engineering/*` (390 lines total) all the scripts of the data engineering side (no data).

I used the following data sources:

- [GEOJson files](https://github.com/jiboncom/chile_geojson)

- [Census data](https://www.ide.cl/index.php/planificacion-y-catastro/item/1948-microdatos-censo-2017-manzana)

- Satellite images: 3.62 GB of satellite images of Santiago can be accessed upon request (for collaborations).

- Map of Santiago with a prediction for each block: https://goyanedelv.github.io/santiago/
