"""
Suffix Cache Server for distributed cache synchronization.

This module provides an HTTP server that accepts requests to update the suffix
cache, enabling synchronization across multiple rollout instances in RL frameworks.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter

logger = logging.getLogger(__name__)


class SuffixCacheRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for suffix cache updates."""

    # Class-level reference to the cache adapter (set during server initialization)
    cache_adapter: Optional["SuffixCacheAdapter"] = None

    def log_message(self, format, *args):
        """Override to use our logger instead of stderr."""
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json_response(self, status_code: int, data: dict):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        """Handle GET requests - health check."""
        if self.path == "/health":
            self._send_json_response(200, {"status": "healthy"})
        elif self.path == "/stats":
            if self.cache_adapter is None:
                self._send_json_response(503, {"error": "Cache adapter not initialized"})
                return
            stats = {
                "active_requests": len(self.cache_adapter.req_state),
                "cached_requests": len(self.cache_adapter.suffix_cache.cached_requests),
            }
            self._send_json_response(200, stats)
        else:
            self._send_json_response(404, {"error": "Not found"})

    def _validate_update_item(self, data: dict):
        if not isinstance(data, dict):
            return False, "item must be a dict"

        if "request_id" not in data:
            return False, "Missing 'request_id' field"
        if "token_ids" not in data:
            return False, "Missing 'token_ids' field"

        request_id = data["request_id"]
        token_ids = data["token_ids"]
        prompt_length = data.get("prompt_length", 0)

        if not isinstance(token_ids, list) or not all(isinstance(t, int) for t in token_ids):
            return False, "'token_ids' must be a list of integers"

        if not isinstance(prompt_length, int) or prompt_length < 0:
            return False, "'prompt_length' must be a non-negative integer"

        if prompt_length > len(token_ids):
            return False, "'prompt_length' cannot exceed total token count"

        return True, (request_id, token_ids, prompt_length)


    def _handle_one_update(self, data: dict):
        ok, parsed_or_error = self._validate_update_item(data)
        if not ok:
            raise ValueError(parsed_or_error)

        request_id, token_ids, prompt_length = parsed_or_error
        prompt = token_ids[:prompt_length]
        response = token_ids[prompt_length:]

        self.cache_adapter.batch_put(
            batch_req_ids=[request_id],
            batch_tokens=[token_ids],
            batch_prompts=[prompt],
        )

        return {
            "status": "success",
            "request_id": request_id,
            "prompt_length": prompt_length,
            "response_length": len(response),
            "total_tokens": len(token_ids),
        }

    def do_POST(self):
        """Handle POST requests - cache updates."""
        if self.path not in ["/update_cache", "/update_cache_batch"]:
            self._send_json_response(404, {"error": "Not found"})
            return

        if self.cache_adapter is None:
            self._send_json_response(503, {"error": "Cache adapter not initialized"})
            return

        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            if self.path == "/update_cache":
                result = self._handle_one_update(data)
                self._send_json_response(200, result)
                return

            # /update_cache_batch
            if not isinstance(data, list):
                self._send_json_response(
                    400, {"error": "Batch update body must be a list"}
                )
                return

            batch_req_ids = []
            batch_tokens = []
            batch_prompts = []
            results = []

            for idx, item in enumerate(data):
                ok, parsed_or_error = self._validate_update_item(item)
                if not ok:
                    self._send_json_response(
                        400,
                        {
                            "error": f"Invalid item at index {idx}: {parsed_or_error}",
                        },
                    )
                    return

                request_id, token_ids, prompt_length = parsed_or_error
                prompt = token_ids[:prompt_length]
                response = token_ids[prompt_length:]

                batch_req_ids.append(request_id)
                batch_tokens.append(token_ids)
                batch_prompts.append(prompt)

                results.append(
                    {
                        "status": "success",
                        "request_id": request_id,
                        "prompt_length": prompt_length,
                        "response_length": len(response),
                        "total_tokens": len(token_ids),
                    }
                )
            
            # logger.info(
            #     "[SuffixCacheRequestHandler::do_POST] "
            #     "tp? cache_adapter_id=%s suffix_cache_id=%s cached_requests_id=%s",
            #     id(self.cache_adapter),
            #     id(self.cache_adapter.suffix_cache),
            #     id(self.cache_adapter.suffix_cache.cached_requests),
            # )

            if batch_req_ids:

                # cached_requests = self.cache_adapter.suffix_cache.cached_requests
                
                # before_keys = set(cached_requests)
                # before_len = len(cached_requests)
                
                self.cache_adapter.batch_put(
                    batch_req_ids=batch_req_ids,
                    batch_tokens=batch_tokens,
                    batch_prompts=batch_prompts,
                )

                # cached_requests = self.cache_adapter.suffix_cache.cached_requests

                # after_keys = set(cached_requests)
                # after_len = len(cached_requests)

                # inserted = [rid for rid in batch_req_ids if rid in after_keys]
                # evicted = before_keys - after_keys

                # logger.info(
                #     "[SuffixCacheRequestHandler::do_POST] update_cache_batch: "
                #     "num_updates=%d, len %d -> %d, delta=%d, "
                #     "inserted=%d, evicted=%d, "
                #     "first_req_id=%s, in_after=%s",
                #     len(batch_req_ids),
                #     before_len,
                #     after_len,
                #     after_len - before_len,
                #     len(inserted),
                #     len(evicted),
                #     batch_req_ids[0] if batch_req_ids else None,
                #     batch_req_ids[0] in after_keys if batch_req_ids else None,
                # )

            self._send_json_response(
                200,
                {
                    "status": "success",
                    "num_updates": len(batch_req_ids),
                    "results": results,
                },
            )

        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "Invalid JSON"})
        except Exception as e:
            logger.exception("Error processing cache update request")
            self._send_json_response(500, {"error": str(e)})


class SuffixCacheServer:
    """
    HTTP server for suffix cache updates.

    This server runs in a separate thread and accepts HTTP requests to update
    the suffix cache. It's useful for synchronizing cache state across multiple
    rollout instances in RL frameworks.

    Usage:
        server = SuffixCacheServer(port=6378, cache_adapter=adapter)
        server.start()
        # ... server is running ...
        server.stop()
    """

    def __init__(
        self,
        port: int,
        cache_adapter: "SuffixCacheAdapter",
        host: str = "0.0.0.0",
    ):
        """
        Initialize the suffix cache server.

        Args:
            port: Port number to listen on
            cache_adapter: The SuffixCacheAdapter instance to update
            host: Host address to bind to (default: "0.0.0.0" for all interfaces)
        """
        self.port = port
        self.host = host
        self.cache_adapter = cache_adapter

        # Set the class-level cache adapter reference
        SuffixCacheRequestHandler.cache_adapter = cache_adapter

        # Server state
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the cache server in a background thread."""
        if self._running:
            logger.warning("SuffixCacheServer is already running")
            return

        try:
            # Create the HTTP server
            self._server = HTTPServer((self.host, self.port), SuffixCacheRequestHandler)

            # Start server in a background thread
            self._running = True
            self._thread = threading.Thread(target=self._run_server, daemon=True)
            self._thread.start()

            logger.info(
                f"SuffixCacheServer started on {self.host}:{self.port}. "
                f"POST /update_cache to update the cache. "
                f"GET /health for health check. "
                f"GET /stats for cache statistics."
            )
        except Exception as e:
            self._running = False
            logger.error(f"Failed to start SuffixCacheServer: {e}")
            raise

    def _run_server(self):
        """Run the HTTP server (called in background thread)."""
        try:
            self._server.serve_forever()
        except Exception as e:
            if self._running:
                logger.error(f"SuffixCacheServer error: {e}")
            self._running = False

    def stop(self):
        """Stop the cache server."""
        if not self._running:
            return

        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            self._thread = None

        logger.info("SuffixCacheServer stopped")

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running

    def __del__(self):
        """Ensure the server is stopped on deletion."""
        self.stop()


# API client for updating remote caches
class SuffixCacheClient:
    """
    Client for updating remote suffix cache servers.

    This client can be used by RL frameworks to broadcast cache updates
    to multiple rollout instances.

    Usage:
        client = SuffixCacheClient(["http://host1:6378", "http://host2:6378"])
        client.update_cache("request_123", [1, 2, 3, 4, 5], prompt_length=2)
    """

    def __init__(self, server_urls: list[str]):
        """
        Initialize the cache client.

        Args:
            server_urls: List of server URLs (e.g., ["http://host1:6378", "http://host2:6378"])
        """
        self.server_urls = server_urls

    def update_cache(
        self, request_id: str, token_ids: list[int], prompt_length: int = 0
    ) -> dict:
        """
        Update the cache on all servers.

        Args:
            request_id: Unique identifier for the request
            token_ids: List of token IDs (prompt + response)
            prompt_length: Length of the prompt portion in token_ids.
                           The remaining tokens are treated as response.
                           Defaults to 0 (all tokens treated as response).

        Returns:
            Dict with results for each server
        """
        import urllib.request
        import urllib.error

        results = {}
        data = json.dumps(
            {
                "request_id": request_id,
                "token_ids": token_ids,
                "prompt_length": prompt_length,
            }
        ).encode("utf-8")

        for url in self.server_urls:
            try:
                req = urllib.request.Request(
                    f"{url.rstrip('/')}/update_cache",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    results[url] = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as e:
                results[url] = {"error": str(e)}
            except Exception as e:
                results[url] = {"error": str(e)}

        return results

    def health_check(self) -> dict:
        """
        Check health of all servers.

        Returns:
            Dict with health status for each server
        """
        import urllib.request
        import urllib.error

        results = {}
        for url in self.server_urls:
            try:
                req = urllib.request.Request(f"{url.rstrip('/')}/health")
                with urllib.request.urlopen(req, timeout=5) as response:
                    results[url] = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as e:
                results[url] = {"error": str(e)}
            except Exception as e:
                results[url] = {"error": str(e)}

        return results
