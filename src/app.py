from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime, timedelta
import asyncio
import pickle
import pandas as pd
import numpy as np
import joblib
from forecast import *
from llm import LLMMarketAnalyst
from loguru import logger

app = FastAPI(
    title="HDB Price Prediction & BTO Recommendation API",
    description="Comprehensive HDB market analysis system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PricePredictionRequest(BaseModel):
    town: str
    flat_type: str
    storey_range: str
    floor_area_sqm: float
    lease_commence_date: int
    
class PricePredictionResponse(BaseModel):
    predicted_price: int
    analysis: str

# Load the model from file
model = joblib.load("./checkpoint/regres_lgb_20250620.txt.pkl")
with open("./checkpoint/regres_lgb_20250620.scale", "rb") as fo:
    scaler, feat_meta, twon_set, flat_set = pickle.load(fo)

agent = LLMMarketAnalyst()

# API Routes
@app.get("/healthcheck")
async def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PricePredictionResponse)
async def predict_price(request: PricePredictionRequest):
    """Predict HDB resale flat price"""
    try:
        # Load model and make prediction
        input_data = pd.DataFrame([request.model_dump()])
        input_data["town_tag"] = input_data["town"].apply(lambda x: twon_set.get(x, None))
        input_data["storey_range"] = input_data["storey_range"].apply(lambda x: floor_cat(x))
        input_data["flat_type_tag"] = input_data["flat_type"].apply(lambda x: flat_set.get(x, None))
        input_data["floor_area_sqm"] = input_data["floor_area_sqm"].apply(np.log)
        input_data["remaining_lease"] = input_data["lease_commence_date"].apply(lambda x: np.log(datetime.now().year-x+1))
        input_data["storey_range"] = input_data["storey_range"].astype("category")
        input_data["town_tag"] = input_data["town_tag"].astype("category")
        input_data["flat_type_tag"] = input_data["flat_type_tag"].astype("category")

        test_x_num = input_data[feat_meta["num"]]
        test_x_scaled_num = scaler.transform(test_x_num)
        test_x_scaled_num = pd.DataFrame(test_x_scaled_num, columns=feat_meta["num"])
        test_x_scaled = pd.concat([test_x_scaled_num, input_data[feat_meta["cat"]].reset_index(drop=True)], axis=1)

        prediction = model.predict(test_x_scaled)

        #agent analysis
        qflat = input_data["flat_type"]
        qtown = input_data["town"]
        dt_first = (datetime.now() - timedelta(days = 365 * 2)).strftime("%Y-%m")
        sql_str = f"""SELECT month, flat_type, storey_range, floor_area_sqm, resale_price FROM hdb_resale_transactions 
        where month>='{dt_first}' AND town='{qtown}' AND flat_type='{qflat}' ORDER BY month ASC;
        """
        logger.info(sql_str)
        price_data = await qstdb.query_data(sql_str)
        comparative_data = ""
        result = agent.analyze_market_trends(qtown, price_data, comparative_data)
        logger.info(result)
                
        return PricePredictionResponse(
            predicted_price = int(np.exp(prediction[0])),
            analysis = result,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)