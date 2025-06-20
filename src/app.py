from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import asyncio

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

# Pydantic models for request/response
class PricePredictionRequest(BaseModel):
    town: str
    flat_type: str
    storey_range: str
    floor_area_sqm: float
    lease_commence_date: int
    
class PricePredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: Dict[str, float]
    price_per_sqm: float
    market_analysis: str
    factors: Dict[str, Any]

class BTORecommendationRequest(BaseModel):
    requirements: str
    budget_range: Optional[Dict[str, float]] = None
    preferred_regions: Optional[List[str]] = None
    flat_types: List[str] = ["3 ROOM", "4 ROOM", "5 ROOM"]
    
class BTORecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    market_insights: str
    pricing_analysis: Dict[str, Any]
    affordability_breakdown: Dict[str, Any]

# API Routes
@app.get("/")
async def root():
    return {"message": "HDB Price Prediction & BTO Recommendation API"}

@app.post("/predict/price", response_model=PricePredictionResponse)
async def predict_price(request: PricePredictionRequest):
    """Predict HDB resale flat price"""
    try:
        # Load model and make prediction
        model = get_price_model()
        llm_analyst = get_llm_analyst()
        
        # Prepare input data
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction with confidence intervals
        prediction, lower_ci, upper_ci = model.predict_with_confidence(input_data)
        
        # Generate market analysis
        market_analysis = llm_analyst.explain_price_prediction(
            prediction[0], 
            model.feature_importance.head(5).to_dict()
        )
        
        return PricePredictionResponse(
            predicted_price=float(prediction[0]),
            confidence_interval={
                "lower": float(lower_ci[0]),
                "upper": float(upper_ci[0])
            },
            price_per_sqm=float(prediction[0] / request.floor_area_sqm),
            market_analysis=market_analysis,
            factors=get_price_factors(request.town, request.flat_type)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/bto", response_model=BTORecommendationResponse)
async def recommend_bto(request: BTORecommendationRequest):
    """Recommend BTO development locations"""
    try:
        # Initialize recommendation engine
        rec_engine = get_bto_engine()
        
        # Generate recommendations
        recommendations = rec_engine.generate_recommendations(request.requirements)
        
        return BTORecommendationResponse(
            recommendations=recommendations['analysis'],
            market_insights=recommendations['recommendations'],
            pricing_analysis=format_pricing_analysis(recommendations),
            affordability_breakdown=calculate_affordability_breakdown(recommendations)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/trends/{town}")
async def get_market_trends(town: str, period: str = "12m"):
    """Get market trends for specific town"""
    try:
        trends_data = get_town_trends(town, period)
        llm_analyst = get_llm_analyst()
        
        analysis = llm_analyst.analyze_market_trends(
            town, 
            trends_data['price_data'],
            trends_data['comparative_data']
        )
        
        return {
            "town": town,
            "period": period,
            "trends": trends_data,
            "analysis": analysis,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/towns")
async def get_all_towns():
    """Get list of all HDB towns with basic info"""
    return get_towns_list()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Background task for model retraining
@app.post("/admin/retrain")
async def trigger_model_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining (admin only)"""
    background_tasks.add_task(retrain_models)
    return {"message": "Model retraining initiated"}

async def retrain_models():
    """Background task to retrain models"""
    # Implementation for model retraining
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)