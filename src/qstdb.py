import requests
import json
import os
from loguru import logger
import sys
from dotenv import load_dotenv
from pathlib import Path
from requests.auth import HTTPBasicAuth
import asyncio
import asyncpg
import pandas as pd
from datetime import datetime, timedelta

rpth = Path(os.path.abspath(__file__)).parent.parent
load_dotenv(str(rpth/'.env'))

def setup_logger(log_name = "./logs_data/audit", level="INFO"):
    """
    Setup logger configuration.
    """
    logger.remove()
    logger.add(
        log_name,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        retention="1 days" 
    )

setup_logger()

def run_questdb_query(sql_query):
    query_params = {'query': sql_query, 'fmt' : 'json'}
    try:
        response = requests.get(os.getenv("QDB_HOST") + '/exec', params = query_params, auth = HTTPBasicAuth(os.getenv("QDB_USER"), os.getenv("QDB_PWD")))
        rsp = json.loads(response.text)
        logger.info(f"{sql_query[:20]} => {response.status_code}")
        return rsp
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error:{e}")
        return None

def query(sql_str):
    async def get_data():
        conn = await asyncpg.connect(
            host = os.getenv("QDB_HOST"),
            port = os.getenv("QDB_PORT_READ"), #8812
            user = os.getenv("QDB_USER"),
            password = os.getenv("QDB_PWD"),
            database = os.getenv("QDB_DB"), #'qdb'
        )
        
        rows = await conn.fetch(sql_str)
        df = pd.DataFrame([dict(row) for row in rows])        
        await conn.close()
        return df

    df = asyncio.run(get_data())

    return df

def execute(sql_str, data = None):
    """
    Execute command. Data is None for command e.g. create, drop
    """
    async def run_cmd():
        conn = await asyncpg.connect(
            host = os.getenv("QDB_HOST"),
            port = os.getenv("QDB_PORT_READ"), #8812
            user = os.getenv("QDB_USER"),
            password = os.getenv("QDB_PWD"),
            database = os.getenv("QDB_DB"), #'qdb'
        )
        if data:
            await conn.executemany(sql_str, data)
        else:
            await conn.execute(sql_str)
        await conn.close()
    try:
        asyncio.run(run_cmd())
        return True
    except Exception as e:
        logger.warning(e)
        return False 

