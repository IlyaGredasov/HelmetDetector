import httpx
import pytest

from server.helmet_server import HelmetServer


class DummyClientSuccess:
    def __init__(self):
        self.called = False
        self.last_payload = None

    async def post(self, url, data, files):
        self.called = True
        self.last_payload = (url, data, files)

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"detection_id": 123}

        return Response()


class DummyClientHttpError:
    def __init__(self):
        self.called = False

    async def post(self, url, data, files):
        self.called = True
        raise httpx.HTTPError("network error")


class DummyClientStatusError:
    def __init__(self):
        self.called = False

    async def post(self, url, data, files):
        self.called = True

        class Response:
            def raise_for_status(self):
                raise httpx.HTTPError("bad status")

            def json(self):
                return {}

        return Response()


@pytest.mark.asyncio
async def test_send_detection_to_api_success():
    server = object.__new__(HelmetServer)
    client = DummyClientSuccess()
    server.http_client = client

    await server.send_detection_to_api(1, "2025-01-01T00:00:00Z", b"data")

    assert client.called
    assert client.last_payload is not None
    url, data, files = client.last_payload
    assert url == "/detections"
    assert data["camera_id"] == "1"
    assert data["detection_time"] == "2025-01-01T00:00:00Z"
    assert files["image"][0] == "frame.jpg"


@pytest.mark.asyncio
async def test_send_detection_to_api_handles_http_error():
    server = object.__new__(HelmetServer)
    client = DummyClientHttpError()
    server.http_client = client

    await server.send_detection_to_api(2, "2025-01-01T00:00:00Z", b"data")

    assert client.called


@pytest.mark.asyncio
async def test_send_detection_to_api_handles_status_error():
    server = object.__new__(HelmetServer)
    client = DummyClientStatusError()
    server.http_client = client

    await server.send_detection_to_api(3, "2025-01-01T00:00:00Z", b"data")

    assert client.called


@pytest.mark.asyncio
async def test_send_detection_to_api_accepts_large_payload():
    server = object.__new__(HelmetServer)
    client = DummyClientSuccess()
    server.http_client = client

    payload = b"x" * 10_000
    await server.send_detection_to_api(4, "2025-01-01T00:00:00Z", payload)

    assert client.called
    assert len(payload) == 10_000


@pytest.mark.asyncio
async def test_send_detection_to_api_multiple_calls():
    server = object.__new__(HelmetServer)
    client = DummyClientSuccess()
    server.http_client = client

    await server.send_detection_to_api(5, "ts1", b"a")
    assert client.called
    client.called = False

    await server.send_detection_to_api(5, "ts2", b"b")
    assert client.called
