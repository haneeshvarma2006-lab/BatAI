"""The UI is one HTML file served at /. This is the whole check."""

import unittest

from tests.test_api import ApiTestCase


class TestUI(ApiTestCase):
    def test_root_serves_the_chat_page_without_auth(self) -> None:
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("/v1/sessions/", r.text)
        self.assertIn("messages/stream", r.text)


if __name__ == "__main__":
    unittest.main()
