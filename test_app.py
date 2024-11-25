import pytest
from fastapi.testclient import TestClient
from tomato_disease_backend import app
import time

client = TestClient(app)

# Test for the prediction endpoint
def test_predict():
    # Prepare a mock image file
    with open("test_image.jpg", "rb") as f:
        response = client.post("/predict/", files={"file": f})
    assert response.status_code == 200
    assert "predicted_label" in response.json()

# # Test for the metrics endpoint
def test_metrics():
     response = client.get("/metrics")
     assert response.status_code == 200
     assert "http_requests_total" in response.text  # Check if the metric is present

def test_invalid_file_type():
    response = client.post("/predict/", files={"file": ("invalid.txt", b"Not an image file")})
    assert response.status_code == 400
    assert "error" in response.json()

def test_missing_file():
    response = client.post("/predict/")
    assert response.status_code == 422  # FastAPI responds with 422 for missing required fields

def test_response_time():
    with open("test_image.jpg", "rb") as f:
        start_time = time.time()
        response = client.post("/predict/", files={"file": f})
        end_time = time.time()
    assert response.status_code == 200
    assert (end_time - start_time) < 3 


