import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LLMMarketAnalyst:
    def __init__(self, api_key):
        self.llm = OpenAI(api_key=api_key, temperature=0.3)
        self.setup_prompt_templates()
    
    def setup_prompt_templates(self):
        """Setup prompt templates for different analysis types"""
        self.price_analysis_template = PromptTemplate(
            input_variables=["town", "price_data", "market_trends"],
            template="""
            As a real estate market analyst, provide a comprehensive analysis of 
            {town}'s HDB resale market:
            
            Price Data: {price_data}
            Market Trends: {market_trends}
            
            Analysis should include:
            1. Current market position
            2. Price trends and drivers
            3. Comparison with similar towns
            4. Investment outlook
            5. Buyer recommendations
            
            Provide specific, data-driven insights in a professional tone.
            """
        )
        
        self.bto_recommendation_template = PromptTemplate(
            input_variables=["requirements", "analysis", "constraints"],
            template="""
            As an urban planning consultant, recommend optimal locations for 
            HDB BTO development based on:
            
            Requirements: {requirements}
            Market Analysis: {analysis}
            Constraints: {constraints}
            
            Provide:
            1. Top 3 recommended locations with detailed justification
            2. Expected pricing for different unit types
            3. Target demographics and affordability analysis
            4. Infrastructure and amenity considerations
            5. Implementation timeline and risks
            
            Format as a structured report with clear recommendations.
            """
        )
    
    def analyze_market_trends(self, town, price_data, comparative_data):
        """Generate market trend analysis"""
        chain = LLMChain(llm=self.llm, prompt=self.price_analysis_template)
        
        analysis = chain.run(
            town=town,
            price_data=price_data,
            market_trends=comparative_data
        )
        
        return analysis
    
    def generate_bto_recommendations(self, requirements, analysis_data):
        """Generate BTO development recommendations"""
        chain = LLMChain(llm=self.llm, prompt=self.bto_recommendation_template)
        
        recommendations = chain.run(
            requirements=requirements,
            analysis=analysis_data,
            constraints="Budget constraints, land availability, infrastructure capacity"
        )
        
        return recommendations
    
    def explain_price_prediction(self, prediction, factors):
        """Explain price prediction with key factors"""
        prompt = f"""
        Explain this HDB resale price prediction in simple terms:
        
        Predicted Price: ${prediction:,.2f}
        Key Factors: {factors}
        
        Provide a clear explanation of:
        1. Why this price is reasonable
        2. Main factors influencing the price
        3. Potential risks or uncertainties
        4. Market context
        
        Keep explanation accessible to general public.
        """
        
        response = self.llm(prompt)
        return response