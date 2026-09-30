"""The paper demo must not expose the removed reference-image method."""
import http.client
import threading
import unittest
from unittest.mock import Mock

import web_demo


class RemovedReferenceTests(unittest.TestCase):
    def test_no_reference_ui_or_endpoint(self):
        self.assertNotIn('reference-file', web_demo.PAGE)
        self.assertNotIn('reference-run', web_demo.PAGE)
        self.assertNotIn('/predict-reference', web_demo.PAGE)
        runtime = Mock(default_size=672, device='cpu')
        server = web_demo.ThreadingHTTPServer(('127.0.0.1', 0), web_demo.make_handler(runtime))
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                connection.request('POST', '/predict-reference', b'')
                response = connection.getresponse()
                self.assertEqual(response.status, 404)
                response.read()
            finally:
                connection.close()
            runtime.predict.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
