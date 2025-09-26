import requests

url = "https://fastapi-ai-service-1081333106174.asia-southeast2.run.app/scan-asuransi"
file_path = "/home/lenovo/UGM/Lomba/COMPFEST/NLXOTI-AI/test_uploadfile/data/ex_asuransi.jpg"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/png")}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())