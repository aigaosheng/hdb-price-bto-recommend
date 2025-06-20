import requests

url = "http://localhost:8000/predict"

payload = {
    "flat_type": "4 ROOM",
    "floor_area_sqm": 93.0,
    "storey_range": "10 TO 12",
    "lease_commence_date": 2005,
    "town": "ANG MO KIO"
}

response = requests.post(url, json=payload)
rsp = response.json()

with open("test_api_output.txt", "wt") as fo:
    fo.write(f"Predicted_price : {rsp['predicted_price']}\n")
    fo.write(f"Analysis\n{rsp['analysis']}")
print(f"predicted_price : {rsp['predicted_price']}")
print(f"Analysis\n{rsp['analysis']}")

