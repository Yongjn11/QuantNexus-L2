"""
QuantNexus-L2 :: News & Chaos Ingestor
======================================

Asynchronous news and event feed manager supporting dual modes:
  1. Live Public RSS Polling: Non-blocking fetching and deduplication from free
     public crypto / macro feeds (CoinDesk, Cointelegraph).
  2. Deterministic Chaos / Black-Swan Injector: In-memory async queue enabling
     tests, simulations, and the Streamlit UI to inject synthetic market-moving
     shocks for immediate evaluation by the Slow-Loop agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import aiohttp

log = logging.getLogger("quantnexus.slow_loop.news")

DEFAULT_RSS_FEEDS: tuple[str, ...] = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)


@dataclass(frozen=True, slots=True)
class NewsEvent:
    headline: str
    source: str = "CHAOS_INJECTOR"
    published_at: Optional[str] = None
    link: Optional[str] = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict:
        return asdict(self)


class NewsAndChaosIngestor:
    """Asynchronous news polling and deterministic chaos event injector."""

    def __init__(
        self,
        rss_feeds: Optional[Sequence[str]] = None,
        timeout_s: float = 5.0,
    ) -> None:
        self.rss_feeds = tuple(rss_feeds or DEFAULT_RSS_FEEDS)
        self.timeout_s = timeout_s
        self._seen_headlines: set[str] = set()
        self._queue: asyncio.Queue[NewsEvent] = asyncio.Queue()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def clear(self) -> None:
        """Clear the seen cache and flush the queue."""
        self._seen_headlines.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ----------------------------------------------------------------------- #
    # Deterministic Chaos / Event Injection
    # ----------------------------------------------------------------------- #
    async def inject_event(
        self, headline: str, source: str = "CHAOS_INJECTOR", link: Optional[str] = None
    ) -> NewsEvent:
        """Inject a synthetic market-moving headline for immediate agent processing."""
        clean = headline.strip()
        event = NewsEvent(headline=clean, source=source, link=link)
        self._seen_headlines.add(clean.lower())
        await self._queue.put(event)
        log.info("Injected event [%s]: %s", source, clean)
        return event

    def inject_event_nowait(
        self, headline: str, source: str = "CHAOS_INJECTOR", link: Optional[str] = None
    ) -> NewsEvent:
        """Synchronous variant of inject_event."""
        clean = headline.strip()
        event = NewsEvent(headline=clean, source=source, link=link)
        self._seen_headlines.add(clean.lower())
        self._queue.put_nowait(event)
        return event

    # ----------------------------------------------------------------------- #
    # Live RSS Polling & Parsing
    # ----------------------------------------------------------------------- #
    async def fetch_feed(
        self, session: aiohttp.ClientSession, url: str
    ) -> list[NewsEvent]:
        """Fetch and parse one RSS feed with error handling and deduplication."""
        new_events: list[NewsEvent] = []
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                headers={"User-Agent": "QuantNexus-L2/1.0 (Institutional Risk Feed)"},
            ) as response:
                if response.status != 200:
                    log.warning("RSS feed %s returned HTTP %s", url, response.status)
                    return new_events

                content = await response.text()
                root = ET.fromstring(content)

                # Channel -> Items
                source_name = "RSS"
                channel_title = root.find(".//channel/title")
                if channel_title is not None and channel_title.text:
                    source_name = channel_title.text.strip()

                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    if title_elem is None or not title_elem.text:
                        continue
                    headline = title_elem.text.strip()
                    key = headline.lower()
                    if key in self._seen_headlines:
                        continue

                    self._seen_headlines.add(key)
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate")
                    link = link_elem.text.strip() if link_elem is not None and link_elem.text else None
                    pub = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else None

                    event = NewsEvent(
                        headline=headline,
                        source=source_name,
                        published_at=pub,
                        link=link,
                    )
                    new_events.append(event)
                    await self._queue.put(event)

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("Network timeout/error fetching RSS %s: %r", url, exc)
        except ET.ParseError as exc:
            log.warning("XML parse error on RSS %s: %r", url, exc)
        except Exception as exc:
            log.warning("Unexpected error polling RSS %s: %r", url, exc)

        return new_events

    async def poll_all_feeds(self) -> list[NewsEvent]:
        """Poll all configured RSS feeds concurrently and queue new headlines."""
        all_new: list[NewsEvent] = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_feed(session, url) for url in self.rss_feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_new.extend(res)
        return all_new

    # ----------------------------------------------------------------------- #
    # Consumption API
    # ----------------------------------------------------------------------- #
    async def get_next_event(self, timeout: Optional[float] = None) -> Optional[NewsEvent]:
        """Retrieve the next queued event, waiting up to timeout seconds."""
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    def get_next_event_nowait(self) -> Optional[NewsEvent]:
        """Retrieve the next queued event immediately, or None if empty."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
