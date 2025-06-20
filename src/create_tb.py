import os
from pathlib import Path
import sys
from questdb.ingress import Sender, TimestampNanos

rpth = Path(os.path.abspath(__file__)).parent.parent/"src"
sys.path.append(str(rpth))
for  v in rpth.iterdir():
    if v.is_dir() and v.name != "__pycache__":
         sys.path.append(str(v))

import qstdb

def create_tb():
    sql_str = """
    -- Core transaction data
    CREATE TABLE IF NOT EXISTS resale_transactions (
        month DATE NOT NULL,
        town VARCHAR NOT NULL,
        flat_type VARCHAR NOT NULL,
        storey_range VARCHAR NOT NULL,
        floor_area_sqm DOUBLE NOT NULL,
        flat_model VARCHAR,
        lease_commence_date INTEGER,
        resale_price DOUBLE NOT NULL,
        price_per_sqm DOUBLE ,
        remaining_lease INTEGER 
    );

    -- Town characteristics and amenities
    CREATE TABLE IF NOT EXISTS town_profiles (
        town VARCHAR,
        town_type VARCHAR, --'mature', 'non_mature'
        region VARCHAR,
        mrt_stations INTEGER ,
        primary_schools INTEGER,
        secondary_schools INTEGER,
        shopping_malls INTEGER,
        hawker_centers INTEGER,
        parks_count INTEGER,
        cbd_distance_km DOUBLE,
        population INTEGER,
        median_household_income DOUBLE,
        updated_at TIMESTAMP
    );

    -- BTO launch history
    CREATE TABLE IF NOT EXISTS bto_launches (
        launch_date DATE,
        town VARCHAR,
        project_name VARCHAR,
        flat_type VARCHAR,
        units_offered INTEGER,
        launch_price_from DOUBLE,
        launch_price_to DOUBLE,
        estimated_completion DATE,
        subscription_rate DOUBLE,
        created_at TIMESTAMP
    );

    -- Price prediction results cache
    CREATE TABLE IF NOT EXISTS price_predictions (
        town VARCHAR,
        flat_type VARCHAR,
        storey_range VARCHAR,
        floor_area_sqm DOUBLE,
        predicted_price DOUBLE,
        confidence_interval_lower DOUBLE,
        confidence_interval_upper DOUBLE,
        model_version VARCHAR,
        prediction_date TIMESTAMP,
        INDEX VARCHAR
    );

    -- Market insights and LLM-generated content
    -- insight_type ('price_trend', 'bto_recommendation', 'market_comparison') NOT NULL,
    CREATE TABLE IF NOT EXISTS market_insights (
        town VARCHAR,
        insight_type VARCHAR, 
        content TEXT,
        metadata VARCHAR,
        generated_at TIMESTAMP
    );

    -- Model performance tracking
    CREATE TABLE IF NOT EXISTS model_metrics (
        model_name VARCHAR,
        model_version VARCHAR,
        metric_name VARCHAR,
        metric_value DOUBLE,
        test_date TIMESTAMP
    );  
    """
    qstdb.execute(sql_str)
      
if __name__ == "__main__":
    create_tb()
