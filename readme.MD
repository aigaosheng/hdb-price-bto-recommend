## HDB Resale Price Prediction & BTO Recommendation System

### Comprehensive Architecture & Implementation Plan

- System Overview:
    - The system combines traditional machine learning with Large Language Models to analyze HDB resale flat prices and recommend optimal BTO development locations. It provides comprehensive market insights, location-specific comparisons, and AI-generated natural language explanations.
- Key Capabilities:

    - Predict resale flat prices across Singapore towns
    - Recommend optimal estates for BTO development
    - Generate price predictions for different flat archetypes
    - Provide natural language market insights and comparisons
    - Match pricing to different income brackets

- Data Engineering & Architecture
    - Data Sources
    - Primary Data Source:

HDB Resale Flat Prices: data.gov.sg API (1990-2025 data available)

- Endpoint: https://data.gov.sg/api/action/datastore_search?resource_id=8b84c4ee58e3cfc0ece0d773c8ca6abc
Fields: month, town, flat_type, storey_range, floor_area_sqm, flat_model, lease_commence_date, resale_price

## Process flow
1. Start questdb_data server
```sudo docker run -p 9000:9000 -p 9009:9009 -p 8812:8812 -v /home/gs/work/db_setup/questdb_data:/var/lib/questdb questdb/questdb```
2. ```python src/create_db.py```
3. Fetch data to DB: ```python src/data_fetch.py```
4. Train & evaluation, save model & result to DB: ```python src/forecast.py```
    - Metric on test set: 
    ```Testid: regres_lgb_20250620, {"mae": 289196.08727202355, "rmse": 318385.6042422002, "r2": -1.9980424481612467}```
5. Rescale price forecasting API: ```python src/app.py```
6. Test endpoint: ```python src/test_endpoint.py```
    - Query example: ```payload = {
    "flat_type": "4 ROOM",
    "floor_area_sqm": 93.0,
    "storey_range": "10 TO 12",
    "lease_commence_date": 2005,
    "town": "ANG MO KIO"
    }```
    - [Query response results](test_api_output.txt)
## Data analysis
- Based transactions: 1990-01 - 2025-06
| mean               | std                | median   | Total  | flat_type        |
| ------------------ | ------------------ | -------- | ------ | ---------------- |
| 169542.35074385512 | 110586.66172077443 | 126464.0 | 12368  | 2 ROOM           |
| 204458.39256100226 | 113574.92127672673 | 171008.0 | 292940 | 3 ROOM           |
| 334457.7789925098  | 157705.08456544662 | 293888.0 | 349383 | 4 ROOM           |
| 437378.0428597073  | 167768.76262035046 | 407552.0 | 193289 | 5 ROOM           |
| 516974.7414570242  | 165890.57753826308 | 487424.0 | 68585  | EXECUTIVE        |
| 66557.71670515781  | 61387.47222138814  | 48896.0  | 1299   | 1 ROOM           |
| 620091.6818181818  | 208098.15243464347 | 540672.0 | 264    | MULTI-GENERATION |
| 453941.17562724015 | 146596.5433985299  | 466944.0 | 279    | MULTI GENERATION |
