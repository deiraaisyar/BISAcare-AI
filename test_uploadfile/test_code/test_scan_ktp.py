import requests

url = "https://fastapi-ai-service-1081333106174.asia-southeast2.run.app/scan-ktp"
file_path = "/home/lenovo/UGM/Lomba/COMPFEST/NLXOTI-AI/test_uploadfile/data/ex_ktp.jpg"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())