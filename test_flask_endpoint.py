import requests

url = "http://127.0.0.1:5000/predict_soil"

# Test 1: Upload person in suit image
with open("dataset/sanity_check_real_non_soil/person_in_suit.jpg", "rb") as f:
    files = {"image": ("person_in_suit.jpg", f, "image/jpeg")}
    resp = requests.post(url, files=files)

print("--- FLASK API ENDPOINT TEST: PERSON IN SUIT ---")
print("HTTP Status Code:", resp.status_code)
print("Response JSON:", resp.json())
