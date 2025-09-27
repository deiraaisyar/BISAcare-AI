import requests

url = "https://fastapi-ai-service-1081333106174.asia-southeast2.run.app/transcribe"
file_path = "/home/lenovo/UGM/Lomba/COMPFEST/NLXOTI-AI/test_uploadfile/data/ex_voice.m4a"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "audio/m4a")}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
try:
    print("Response:", response.json())
except Exception:
    print("Raw Response:", response.text)
