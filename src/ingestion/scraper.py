"""Async Playwright ingestion with response capture and session reuse.

The scraper deliberately does not bypass CAPTCHA challenges.  It detects a
challenge, exposes its audio URL to an explicitly supplied local solver, or
injects a caller-provided token after the caller has obtained one.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AudioSolver = Callable[[str], str | None | Awaitable[str | None]]


@dataclass(slots=True)
class ScraperConfig:
    """Browser and scraping options."""

    timeout_ms: int = 30_000
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    )
    storage_state_path: str | Path | None = None
    headless: bool = True
    max_concurrency: int = 4
    captcha_token: str | None = None


@dataclass(slots=True)
class CaptchaChallenge:
    """A detected CAPTCHA challenge and optional audio resource."""

    provider: str
    audio_url: str | None = None
    token: str | None = None


@dataclass(slots=True)
class PlaywrightScraper:
    """Capture JSON responses from JavaScript-rendered pages."""

    config: ScraperConfig = field(default_factory=ScraperConfig)
    audio_solver: AudioSolver | None = None
    _browser: Any = field(default=None, init=False, repr=False)
    _context: Any = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> PlaywrightScraper:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for browser ingestion; install 'playwright'."
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.config.headless)
        context_kwargs: dict[str, Any] = {
            "user_agent": self.config.user_agent,
            "viewport": {"width": 1440, "height": 900},
        }
        state_path = self.config.storage_state_path
        if state_path and Path(state_path).exists():
            context_kwargs["storage_state"] = str(state_path)
        self._context = await self._browser.new_context(**context_kwargs)
        self._context.set_default_timeout(self.config.timeout_ms)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._context is not None and self.config.storage_state_path:
            state_path = Path(self.config.storage_state_path)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=str(state_path))
        if self._browser is not None:
            await self._browser.close()
        if hasattr(self, "_playwright"):
            await self._playwright.stop()

    async def scrape_json(
        self,
        url: str,
        *,
        wait_for: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Navigate to ``url`` and return JSON response bodies.

        Only responses whose content type is JSON (or whose URL ends in
        ``.json``) are captured.  A semaphore bounds concurrent page work.
        """
        if self._context is None:
            raise RuntimeError("PlaywrightScraper must be used as an async context manager")
        page = await self._context.new_page()
        captured: list[dict[str, Any]] = []

        if headers:
            await page.set_extra_http_headers(dict(headers))

        async def capture(response: Any) -> None:
            content_type = (response.headers.get("content-type") or "").lower()
            if "json" not in content_type and not response.url.lower().endswith(".json"):
                return
            try:
                payload = await response.json()
            except (TypeError, ValueError):
                return
            if isinstance(payload, dict):
                captured.append(payload)
            else:
                captured.append({"data": payload})

        page.on("response", capture)
        try:
            await page.goto(url, wait_until="networkidle", timeout=self.config.timeout_ms)
            if wait_for:
                await page.wait_for_selector(wait_for)
            # CAPTCHA handling must never block the orchestration loop: without
            # a token/solver the challenge is logged and scraping continues
            # with whatever JSON was captured before the challenge appeared.
            challenge = await self.detect_captcha(page)
            if challenge is not None:
                try:
                    await self.handle_captcha(page, challenge)
                except RuntimeError as exc:
                    logger.warning(
                        "captcha_unsolved_url_continued", extra={"url": url, "error": str(exc)}
                    )
            await asyncio.sleep(0)
            return captured
        except Exception as exc:  # noqa: BLE001 - a single bad page must not crash callers
            logger.warning("scrape_json_failed", extra={"url": url, "error": str(exc)})
            return captured
        finally:
            await page.close()

    async def scrape_page(
        self,
        url: str,
        *,
        wait_for: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Capture response bodies from a JavaScript-rendered page.

        The returned bodies are intentionally raw bytes so callers can pass
        CSV, JSON, or Parquet responses through their own ingestion pipeline.
        CAPTCHA handling and storage-state reuse follow :meth:`scrape_json`.
        """
        if self._context is None:
            raise RuntimeError("PlaywrightScraper must be used as an async context manager")
        page = await self._context.new_page()
        captured: list[dict[str, Any]] = []

        if headers:
            await page.set_extra_http_headers(dict(headers))

        async def capture(response: Any) -> None:
            content_type = (response.headers.get("content-type") or "").lower()
            try:
                body = await response.body()
            except Exception as exc:  # noqa: BLE001 - an individual response is optional
                logger.warning(
                    "scrape_page_response_failed",
                    extra={"url": response.url, "error": str(exc)},
                )
                return
            captured.append(
                {"url": response.url, "content_type": content_type, "body": body}
            )

        page.on("response", capture)
        try:
            await page.goto(url, wait_until="networkidle", timeout=self.config.timeout_ms)
            if wait_for:
                await page.wait_for_selector(wait_for)
            challenge = await self.detect_captcha(page)
            if challenge is not None:
                try:
                    await self.handle_captcha(page, challenge)
                except RuntimeError as exc:
                    logger.warning(
                        "captcha_unsolved_url_continued", extra={"url": url, "error": str(exc)}
                    )
            await asyncio.sleep(0)
            return captured
        except Exception as exc:  # noqa: BLE001 - a single bad page must not crash callers
            logger.warning("scrape_page_failed", extra={"url": url, "error": str(exc)})
            return captured
        finally:
            await page.close()

    async def detect_captcha(self, page: Any) -> CaptchaChallenge | None:
        """Detect common CAPTCHA markers and extract an audio challenge URL."""
        content = (await page.content()).lower()
        provider = (
            "reCAPTCHA" if "recaptcha" in content else "hCAPTCHA" if "hcaptcha" in content else ""
        )
        if not provider:
            return None
        audio_match = re.search(r"""(?:audio|download)\D{0,80}(https?://[^"' ]+)""", content)
        return CaptchaChallenge(
            provider=provider,
            audio_url=audio_match.group(1) if audio_match else None,
        )

    async def handle_captcha(self, page: Any, challenge: CaptchaChallenge) -> None:
        """Use a supplied local solver or token; otherwise fail explicitly."""
        token = self.config.captcha_token
        if token is None and challenge.audio_url and self.audio_solver is not None:
            token = self.audio_solver(challenge.audio_url)
            if inspect.isawaitable(token):
                token = await token
        if not token:
            raise RuntimeError(
                f"{challenge.provider} detected; provide captcha_token or a local audio_solver"
            )
        challenge.token = token
        await page.evaluate(
            """(value) => {
                for (const selector of ['textarea[name="g-recaptcha-response"]',
                    'textarea[name="h-captcha-response"]']) {
                    const field = document.querySelector(selector);
                    if (field) { field.value = value; field.dispatchEvent(new Event('input')); }
                }
            }""",
            token,
        )


async def scrape_urls(
    urls: list[str],
    *,
    config: ScraperConfig | None = None,
    audio_solver: AudioSolver | None = None,
) -> list[dict[str, Any]]:
    """Scrape URLs with bounded concurrency.

    Failures are isolated per URL: a broken page logs a warning and yields no
    records instead of aborting the whole batch (crash-resilient guarantee).
    """
    effective_config = config or ScraperConfig()
    semaphore = asyncio.Semaphore(effective_config.max_concurrency)
    async with PlaywrightScraper(effective_config, audio_solver) as scraper:

        async def scrape(url: str) -> list[dict[str, Any]]:
            async with semaphore:
                return await scraper.scrape_json(url)

        results = await asyncio.gather(*(scrape(url) for url in urls), return_exceptions=True)

    records: list[dict[str, Any]] = []
    for url, result in zip(urls, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning("scrape_urls_item_failed", extra={"url": url, "error": str(result)})
            continue
        records.extend(result)
    return records


def write_jsonl(records: list[Mapping[str, Any]], path: str | Path) -> Path:
    """Persist captured records as UTF-8 JSON Lines."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), default=str) + "\n")
    return output
