# DSDE Project with Scopus datadset

## Project Structure
```
dsde-proj
|
|-- clean.ipynb                    (load, cleaning)
|-- scopus.ipynb                   (scrape 2024 scopus data)
|-- selenium_scraping.ipynb        (scrape coord, ref/cited count, ASJC code)
|-- viz.py                         (visualization)
|-- mlModel8.py                    (ML)
|-- Kafka/
|    |-- 10122024.csv
|    |-- consumerDaily.py
|    |-- producerDaily.py
|    |-- docker-compose.yml
|-- MLminio/                       (ML flow)
|    |-- Dockerfile
|    |-- ModelMLflow.py
|    |-- docker-compose.yml
|    |-- requirements.txt
|-- .streamlit/
|    |-- config.toml
|-- data/
|    |-- ASJC_cat.csv              (ASJC code)
|    |-- coordinate_country.csv    (Coordinate of affiliations)
|    |-- ref_cite_count.csv        (ref/cited count)
|    |-- ref_cite_count_href.csv   (url to scrape ref/cited count)
|    |-- raw_data.csv              (given scopus data)
|    |-- scopus_data.csv           (scraped 2024 scopus data)
|    |-- viz_data.parquet.gzip     (visualization data)
```


## How to get "raw_data.csv"
1. In "clean.ipynb"
2. run first cell to import library (assuming you already have scopus dataset in your machine)
3. run cell below one of these header __"Load json (multithread)"__ , __"Load json (serial)"__
4. run cell below __"Load json (multithread)"__ to export csv

## To see a visualization
At repo directory
```
streamlit run viz.py
```
