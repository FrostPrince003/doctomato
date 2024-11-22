import pytest
from fastapi.testclient import TestClient
from tomato_disease_backend import app


client = TestClient(app)

# Test for the prediction endpoint
def test_predict():
    # Prepare a mock image file
    with open("test_image.jpg", "rb") as f:
        response = client.post("/predict/", files={"file": f})
    assert response.status_code == 200
    assert "predicted_label" in response.json()

# # Test for the metrics endpoint
# def test_metrics():
#     response = client.get("/metrics")
#     assert response.status_code == 200
#     assert "http_requests_total" in response.text  # Check if the metric is present
