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
print(response.json())
