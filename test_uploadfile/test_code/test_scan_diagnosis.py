import requests

url = "http://localhost:8000/scan-diagnosis"
file_path = "/home/lenovo/UGM/Lomba/COMPFEST/NLXOTI-AI/test_uploadfile/data/ex_diagnosis.jpg"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())