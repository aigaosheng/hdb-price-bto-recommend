import streamlit as st
import requests
import qstdb

# Define the FastAPI endpoint
API_URL = "http://localhost:8000/predict"  # Change to your server address if deployed remotely

# Define dropdown options (you can load these dynamically from the backend too if needed)
TOWN_OPTIONS = qstdb.query("SELECT DISTINCT(town) FROM hdb_resale_transactions")['town'].to_list()
FLAT_TYPE_OPTIONS = qstdb.query("SELECT DISTINCT(flat_type) FROM hdb_resale_transactions")['flat_type'].to_list()
STOREY_OPTIONS = qstdb.query("SELECT DISTINCT(storey_range) FROM hdb_resale_transactions")['storey_range'].to_list()
# TOWN_OPTIONS = [
#     "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", 
#     "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
#     "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
#     "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
#     "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
#     "TOA PAYOH", "WOODLANDS", "YISHUN"
# ]

# FLAT_TYPE_OPTIONS = [
#     "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"
# ]

# STOREY_OPTIONS = [
#     "01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15",
#     "16 TO 18", "19 TO 21", "22 TO 24", "25 TO 27", "28 TO 30",
#     "31 TO 33", "34 TO 36", "37 TO 39", "40 TO 42", "43 TO 45"
# ]

# UI
st.title("🏠 HDB Resale Price Predictor & Market Analyst")

with st.form("price_form"):
    town = st.selectbox("Town", TOWN_OPTIONS)
    flat_type = st.selectbox("Flat Type", FLAT_TYPE_OPTIONS)
    storey_range = st.selectbox("Storey Range", STOREY_OPTIONS)
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=200.0, value=70.0)
    lease_commence_date = st.number_input("Lease Commence Year", min_value=1960, max_value=2025, value=2010)
    
    submitted = st.form_submit_button("Predict Price")

# Form submission
if submitted:
    with st.spinner("Calling model and analyzing market trends..."):
        try:
            response = requests.post(API_URL, json={
                "town": town,
                "flat_type": flat_type,
                "storey_range": storey_range,
                "floor_area_sqm": floor_area_sqm,
                "lease_commence_date": lease_commence_date,
            })
            if response.status_code == 200:
                result = response.json()
                st.success(f"💰 Predicted Resale Price: **${result['predicted_price']:,}**")
                st.markdown("### 📊 Market Analysis")
                st.write(result["analysis"])
            else:
                st.error(f"❌ Prediction failed: {response.json()['detail']}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
