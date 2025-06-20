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
5. Rescale price forecasting API: ```python src/app.py```
6. Test endpoint: ```python src/test_endpoint.py```
