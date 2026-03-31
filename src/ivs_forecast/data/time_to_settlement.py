from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

EASTERN_TZ = ZoneInfo("America/New_York")
NORMAL_SNAPSHOT_TIME = time(hour=15, minute=45)
EARLY_CLOSE_SNAPSHOT_TIME = time(hour=12, minute=45)
SPX_SETTLEMENT_TIME = time(hour=9, minute=30)


def snapshot_timestamp_eastern(quote_date: date, is_early_close: bool) -> datetime:
    snapshot_time = EARLY_CLOSE_SNAPSHOT_TIME if is_early_close else NORMAL_SNAPSHOT_TIME
    return datetime.combine(quote_date, snapshot_time, tzinfo=EASTERN_TZ)


def settlement_timestamp_eastern(expiration: date, option_root: str) -> datetime:
    if option_root != "SPX":
        raise ValueError(
            f"Unsupported option root for settlement-time logic: {option_root}. "
            "Only SPX is supported in v1."
        )
    return datetime.combine(expiration, SPX_SETTLEMENT_TIME, tzinfo=EASTERN_TZ)


def year_fraction_act365(start_ts: datetime, end_ts: datetime) -> float:
    if end_ts <= start_ts:
        raise ValueError(
            f"Settlement timestamp must be later than snapshot timestamp, got {start_ts} -> {end_ts}."
        )
    return (end_ts - start_ts).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)
