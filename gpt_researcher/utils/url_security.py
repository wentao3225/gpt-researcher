"""GPT Researcher 的 URL 安全工具。

当 URL 来自不可信输入（例如 API 和 WebSocket 端点接受的
``source_urls`` / ``document_urls`` 参数）时，这些辅助函数可保护
抓取和文档加载管线免受服务端请求伪造（SSRF）和任意本地文件读取。

默认只允许主机解析为公网 IP 地址的 ``http`` / ``https`` URL。这会阻止：

* 非 HTTP 协议，如 ``file://``、``ftp://`` 和 ``gopher://``。
* 本地文件系统路径（无协议/主机），某些文档加载器否则会直接从磁盘
  读取（任意本地文件读取）。
* 对私有、回环、链路本地、保留或其他内部地址的请求
  （如 ``127.0.0.1``、``10.0.0.0/8`` 以及 ``169.254.169.254``
  云元数据端点）。

有意抓取内部或自托管资源的运维人员，可以通过设置环境变量
``ALLOW_PRIVATE_URLS=true``（或传入 ``allow_private=True``）
来跳过私有地址检查。

注意：此处解析主机后由 HTTP 客户端稍后重新解析，会留下一个小的
TOCTOU/DNS 重绑定窗口。要完全消除它需要将校验过的 IP 固定到连接中；
此防护旨在作为纵深防御，消除最简单的、无需认证的 SSRF
和本地文件读取原语。
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https")

_TRUTHY = ("1", "true", "yes", "on")


class UnsafeURLError(ValueError):
    """当 URL 被 SSRF / 本地文件防护拒绝时抛出。"""


def _private_urls_allowed() -> bool:
    return os.getenv("ALLOW_PRIVATE_URLS", "").strip().lower() in _TRUTHY


def _is_disallowed_ip(ip) -> bool:
    """如果该地址不安全（即内部地址）则返回 True。"""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_url(url: str, *, allow_private: bool | None = None) -> str:
    """校验 ``url`` 是否可以安全抓取，并原样返回。

    参数：
        url: 要校验的 URL。
        allow_private: 为 ``True`` 时跳过私有/内部地址检查。
            为 ``None``（默认）时回退到 ``ALLOW_PRIVATE_URLS``
            环境变量。

    返回：
        通过所有检查时返回原始 ``url``。

    抛出：
        UnsafeURLError: 如果 URL 使用了不允许的协议、缺少主机，
            或解析到非公网地址。
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL must be a non-empty string.")

    parsed = urlparse(url.strip())

    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"URL scheme {scheme or '(none)'!r} is not allowed; "
            "only http and https URLs may be fetched."
        )

    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL must include a valid host.")

    if allow_private is None:
        allow_private = _private_urls_allowed()
    if allow_private:
        return url

    try:
        addrinfo = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host {host!r}: {exc}") from exc

    for info in addrinfo:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise UnsafeURLError(
                f"Host {host!r} resolved to an invalid address {ip_str!r}."
            ) from exc
        if _is_disallowed_ip(ip):
            raise UnsafeURLError(
                f"URL host {host!r} resolves to a non-public address ({ip_str}); "
                "set ALLOW_PRIVATE_URLS=true to allow internal targets."
            )

    return url


def is_safe_url(url: str, *, allow_private: bool | None = None) -> bool:
    """如果 ``url`` 通过 :func:`validate_url` 则返回 ``True``，否则返回 ``False``。"""
    try:
        validate_url(url, allow_private=allow_private)
        return True
    except UnsafeURLError:
        return False
