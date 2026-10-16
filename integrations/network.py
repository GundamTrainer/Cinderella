from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .policies import NetworkPolicy, ProviderPolicyError


class ProviderRequestError(RuntimeError):
    """A bounded provider request failed or returned an invalid response."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class JsonResponse:
    data: dict[str, object]
    final_url: str


class JsonTransport(Protocol):
    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> JsonResponse: ...


def _ensure_public_host(url: str) -> None:
    host = urllib.parse.urlsplit(url).hostname
    if host is None:
        raise ProviderPolicyError("Provider URL is missing a hostname.")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [
                ipaddress.ip_address(info[4][0])
                for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            ]
        except OSError as exc:
            raise ProviderRequestError(f"Could not resolve provider host {host!r}.") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ProviderPolicyError("Provider URL resolves to a non-public IP address.")


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = 3

    def __init__(self, policy: NetworkPolicy) -> None:
        super().__init__()
        self._policy = policy

    def redirect_request(
        self,
        request: urllib.request.Request,
        response: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> urllib.request.Request | None:
        self._policy.validate_url(new_url)
        _ensure_public_host(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


class UrllibJsonTransport:
    """GET-only JSON transport with an exact host allowlist and bounded response reads."""

    def __init__(self, policy: NetworkPolicy) -> None:
        self._policy = policy
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _AllowlistedRedirectHandler(policy),
        )

    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> JsonResponse:
        self._policy.validate_url(url)
        _ensure_public_host(url)
        request_headers = {
            "Accept": "application/json",
            "User-Agent": self._policy.user_agent,
        }
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        try:
            with self._opener.open(request, timeout=self._policy.request_timeout_seconds) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json" and not content_type.endswith("+json"):
                    raise ProviderRequestError(
                        f"Provider returned unsupported content type {content_type!r}."
                    )
                body = response.read(self._policy.maximum_response_bytes + 1)
                if len(body) > self._policy.maximum_response_bytes:
                    raise ProviderRequestError("Provider response exceeded the configured size limit.")
                try:
                    data = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProviderRequestError("Provider returned invalid JSON.") from exc
                if not isinstance(data, dict):
                    raise ProviderRequestError("Provider JSON response must be an object.")
                final_url = response.geturl()
                self._policy.validate_url(final_url)
                return JsonResponse(data=data, final_url=final_url)
        except urllib.error.HTTPError as exc:
            retryable = 500 <= exc.code <= 599
            raise ProviderRequestError(
                f"Provider returned HTTP status {exc.code}.",
                retryable=retryable,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(
                "Provider request failed or timed out.",
                retryable=True,
            ) from exc
