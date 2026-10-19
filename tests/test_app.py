from app import app


def test_health_page_is_available() -> None:
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert "ML Model API is running" in response.get_data(as_text=True)
