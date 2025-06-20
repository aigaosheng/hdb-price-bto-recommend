from langchain_ollama.llms import OllamaLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import qstdb
from datetime import datetime, timedelta

class LLMMarketAnalyst:
    def __init__(self, model_name="gemma3"):
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
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
            
    def analyze_market_trends(self, town, price_data, comparative_data):
        """Generate market trend analysis"""
        chain = LLMChain(llm=self.llm, prompt=self.price_analysis_template)
        return chain.run(
            town=town,
            price_data=price_data,
            market_trends=comparative_data
        )
    
if __name__ == "__main__":
    agent = LLMMarketAnalyst()
    town = "ANG MO KIO"
    dt_first = (datetime.now() - timedelta(days = 365 * 2)).strftime("%Y-%m")
    sql_str = f"""SELECT month, flat_type, storey_range, floor_area_sqm, resale_price FROM hdb_resale_transactions 
    where month>='{dt_first}' AND town='{town}' ORDER BY month ASC;
    """
    price_data = qstdb.query(sql_str)
    comparative_data = ""
    result = agent.analyze_market_trends(town, price_data, comparative_data)
    print(result)