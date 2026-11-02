"""Exercise SDK tool dispatch as well as the local control transport."""
import asyncio
import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional MCP SDK not installed')
class McpBridgeTests(unittest.TestCase):
    def test_sdk_page_navigation_forwards_index_and_filename(self):
        import solid_inpaint_mcp as bridge

        async def check():
            with patch.object(bridge, 'request_control', return_value={'current_page': '2.jpg'}) as call:
                await bridge.mcp.call_tool('go_to_page', {'index': 2})
                call.assert_called_once_with('go_to_page', {'index': 2, 'name': None})
            with patch.object(bridge, 'request_control', return_value={'current_page': '2.jpg'}) as call:
                await bridge.mcp.call_tool('go_to_page', {'name': '2.jpg'})
                call.assert_called_once_with('go_to_page', {'index': None, 'name': '2.jpg'})

        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
