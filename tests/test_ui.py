"""The UI is three static files. Check they're served and wired to the API."""

import unittest

from tests.test_api import ApiTestCase


class TestUI(ApiTestCase):
    def test_root_serves_the_page_without_auth(self) -> None:
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("/static/app.js", r.text)

    def test_assets_are_served(self) -> None:
        js = self.client.get("/static/app.js")
        css = self.client.get("/static/app.css")
        self.assertEqual((js.status_code, css.status_code), (200, 200))
        for route in ("/v1/whoami", "/v1/sessions", "messages/stream", "/v1/memory/documents"):
            self.assertIn(route, js.text)




class TestToolActivityPersists(ApiTestCase):
    """A reloaded chat must show what the agent did, not only what it said."""

    def test_streamed_tool_calls_come_back_in_history(self) -> None:
        from bat.ports.agent import FinalEvent, ToolCallEvent, ToolResultEvent
        from tests.test_api import ACME_KEY, auth

        class ToolRunner:
            async def run(self, request):
                yield ToolCallEvent(call_id="c1", name="calculator", arguments={"expression": "6*7"})
                yield ToolResultEvent(call_id="c1", name="calculator", content="6*7 = 42")
                yield FinalEvent(content="It's 42.")

        self.client.app.state.agent_runner = ToolRunner()
        sid = self.open_session()
        with self.client.stream(
            "POST", f"/v1/sessions/{sid}/messages/stream",
            json={"content": "6*7?"}, headers=auth(ACME_KEY),
        ) as r:
            "".join(r.iter_text())

        items = self.client.get(f"/v1/sessions/{sid}/messages", headers=auth(ACME_KEY)).json()["items"]
        reply = items[-1]
        self.assertEqual(reply["content"], "It's 42.")
        self.assertEqual(reply["tools"][0]["name"], "calculator")
        self.assertEqual(reply["tools"][0]["result"], "6*7 = 42")
        self.assertFalse(reply["tools"][0]["is_error"])
