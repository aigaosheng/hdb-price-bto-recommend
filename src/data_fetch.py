# Data ingestion pipeline (Apache Airflow DAGs)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
from questdb.ingress import Sender, TimestampNanos
import requests
import pandas as pd

rpth = Path(os.path.abspath(__file__)).parent.parent/"src"
sys.path.append(str(rpth))
for  v in rpth.iterdir():
    if v.is_dir() and v.name != "__pycache__":
         sys.path.append(str(v))
import qstdb
import time
import json

IS_LOCAL_TEST = True

def download_file(DATASET_ID):
    # initiate download
    s = requests.Session()
    s.headers.update({'referer': 'https://colab.research.google.com'})
    base_url = "https://api-production.data.gov.sg"
    url = base_url + f"/v2/public/api/datasets/{DATASET_ID}/metadata"
    # print(url)
    response = s.get(url)
    data = response.json()['data']
    columnMetadata = data.pop('columnMetadata', None)

    initiate_download_response = s.get(
        f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download",
        headers={"Content-Type":"application/json"},
        json={}
    )
    print(initiate_download_response.json()['data']['message'])

    # poll download
    MAX_POLLS = 5
    for i in range(MAX_POLLS):
        poll_download_response = s.get(
            f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download",
            headers={"Content-Type":"application/json"},
            json={}
        )
        print("Poll download response:", poll_download_response.json())
        if "url" in poll_download_response.json()['data']:
            print(poll_download_response.json()['data']['url'])
            DOWNLOAD_URL = poll_download_response.json()['data']['url']
            df = pd.read_csv(DOWNLOAD_URL)
            print("\nDataframe loaded!")
            return df
        if i == MAX_POLLS - 1:
            print(f"{i+1}/{MAX_POLLS}: No result found, possible error with dataset, please try again or let us know at https://go.gov.sg/datagov-supportform\n")
        else:
            print(f"{i+1}/{MAX_POLLS}: No result yet, continuing to poll\n")
        time.sleep(3)

def extract_hdb_data():
    """Extract latest HDB resale data from data.gov.sg API"""
    hdb_resale_price_tb = "hdb_resale_transactions"

    conf = f"""http::addr={os.getenv("QDB_HOST")}:{os.getenv("QDB_PORT")};username={os.getenv("QDB_USER")};password={os.getenv("QDB_PWD")};"""
    # Get data ids from HDB resale metadata     
    collection_id = 189 
    try:         
        url_metada = "https://api-production.data.gov.sg/v2/public/api/collections/{}/metadata".format(collection_id)            
        response = requests.get(url_metada)
        metadata = response.json()
    except Exception as e:
        raise Exception(f"** Fetch collection metadat fail -> {e}")
    
    with Sender.from_conf(conf) as sender:
        for dataset_id in metadata["data"]["collectionMetadata"]["childDatasets"]:
            # dataset_id = "d_ebc5ab87086db484f88045b47411ebc5"
            try:
                data = download_file(dataset_id)
                # url_data = "https://data.gov.sg/api/action/datastore_search?resource_id="  + dataset_id            
                # response = requests.get(url_data)
                # data = response.json()
                # data = pd.DataFrame(data['result']['records'])
                data["dataset_id"] = dataset_id
                print(f"** fetch dataset_id => {dataset_id}, {data.shape}")
                sender.dataframe(data, table_name = hdb_resale_price_tb, at = TimestampNanos.now())
                sender.flush() 
                time.sleep(2)
            except Exception as e:
                # print(f"** Fetch transaction dataset_id = {dataset_id} fail -> {e}")
                continue
            

        
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
        extract_hdb_data()