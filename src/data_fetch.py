# Data ingestion pipeline (Apache Airflow DAGs)
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

IS_LOCAL_TEST = True

def extract_hdb_data():
    """Extract latest HDB resale data from data.gov.sg API"""
    import requests
    import pandas as pd
    
    # Extract resale data
    base_url = "https://data.gov.sg/api/action/datastore_search"
    resource_id = "8b84c4ee58e3cfc0ece0d773c8ca6abc"
    
    all_records = []
    offset = 0
    limit = 10000
    
    while True:
        params = {
            'resource_id': resource_id,
            'limit': limit,
            'offset': offset
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        
        records = data['result']['records']
        if not records:
            break
            
        all_records.extend(records)
        offset += limit
    
    df = pd.DataFrame(all_records)
    return df

def enrich_with_amenities():
    """Enrich town data with amenities from OneMap API"""
    # Implementation for amenities data enrichment
    pass

def load_to_database():
    """Load processed data to PostgreSQL"""
    # Implementation for database loading
    pass

if not IS_LOCAL_TEST:
    # DAG definition
    default_args = {
        'owner': 'hdb-system',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': True,
        'retries': 2,
        'retry_delay': timedelta(minutes=5)
    }

    dag = DAG(
        'hdb_data_pipeline',
        default_args=default_args,
        description='HDB data extraction and processing pipeline',
        schedule_interval='@daily',
        catchup=False
    )

    extract_task = PythonOperator(
        task_id='extract_hdb_data',
        python_callable=extract_hdb_data,
        dag=dag
    )

    enrich_task = PythonOperator(
        task_id='enrich_amenities',
        python_callable=enrich_with_amenities,
        dag=dag
    )

    load_task = PythonOperator(
        task_id='load_to_database',
        python_callable=load_to_database,
        dag=dag
    )

    extract_task >> enrich_task >> load_task

else:
    if __name__ == "__main__":
        pass