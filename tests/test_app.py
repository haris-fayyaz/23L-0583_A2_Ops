import io

JPEG = b"\xff\xd8" + b"fake-jpeg-bytes"


def post(client, name, data):
    return client.post("/predict", data={"file": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_upload_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"VisionGuard" in r.data


def test_unknown_page_gives_404(client):
    assert client.get("/nope").status_code == 404


def test_wrong_type_is_415(client):
    assert post(client, "notes.txt", b"hello").status_code == 415


def test_undecodable_image_is_422(client):
    assert post(client, "fake.jpg", b"not an image").status_code == 422


def test_too_large_is_413(client):
    assert post(client, "big.jpg", JPEG + bytes(11 * 1024 * 1024)).status_code == 413


def test_prediction_is_saved_to_history(client):
    r = post(client, "saved.jpg", JPEG)
    assert r.status_code == 200
    assert r.get_json()["label"] == "AI-generated"
    page = client.get("/history").get_data(as_text=True)
    assert "saved.jpg" in page and "stub" in page


def test_rejected_file_is_not_saved(client):
    post(client, "rejected.txt", b"hello")
    assert "rejected.txt" not in client.get("/history").get_data(as_text=True)
