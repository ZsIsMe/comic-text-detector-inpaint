"""Exercise SDK tool dispatch as well as the local control transport."""
import asyncio
import importlib.util
import inspect
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
            with patch.object(bridge, 'request_control', return_value={'current_page': '2.jpg'}) as call:
                await bridge.mcp.call_tool('go_to_page', {'index': 2, 'instance_id': 'a' * 32})
                call.assert_called_once_with('go_to_page', {'index': 2, 'name': None},
                                             instance_id='a' * 32)

        asyncio.run(check())

    def test_every_window_tool_accepts_instance_id_and_list_is_global(self):
        import solid_inpaint_mcp as bridge

        tool_names = ('get_app_status', 'open_project', 'get_detection_settings',
                      'start_detection', 'get_job_status', 'list_pages', 'go_to_page',
                      'get_page_view', 'get_review_bundle', 'apply_page_edits',
                      'undo_page_edit', 'mark_page_reviewed', 'save_project')
        for name in tool_names:
            with self.subTest(name=name):
                self.assertIn('instance_id', inspect.signature(getattr(bridge, name)).parameters)
        with patch.object(bridge, 'discover_app_instances',
                          return_value={'instances': [{'instance_id': 'a' * 32}]}) as discover:
            self.assertEqual(bridge.list_app_instances()['instances'][0]['instance_id'], 'a' * 32)
            discover.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
