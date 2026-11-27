"""Check the provider-imposed wait without using the network."""

import unittest
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

from gemini import quota_retry_delay


class QuotaTests(unittest.TestCase):
    def response(self, payload: bytes, retry_after: str = "") -> HTTPError:
        headers = Message()
        if retry_after:
            headers["Retry-After"] = retry_after
        return HTTPError("https://example.invalid", 429, "limit", headers, BytesIO(payload))

    def test_honors_provider_wait(self):
        error = self.response(b'{"error":{"details":[{"retryDelay":"47.5s"}]}}', "25")
        self.assertEqual(quota_retry_delay(error, 0), 48.5)

    def test_daily_quota_stops_retries(self):
        error = self.response(b'{"error":{"details":[{"violations":[{"quotaId":"RequestsPerDayPerProject"}]}]}}')
        self.assertIsNone(quota_retry_delay(error, 0))


if __name__ == "__main__":
    unittest.main()
