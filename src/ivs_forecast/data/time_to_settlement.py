from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from ivs_forecast.config import SettlementConfig

EASTERN_TZ = ZoneInfo("America/New_York")
NORMAL_SNAPSHOT_TIME = time(hour=15, minute=45)
EARLY_CLOSE_SNAPSHOT_TIME = time(hour=12, minute=45)


def snapshot_timestamp_eastern(quote_date: date, is_early_close: bool) -> datetime:
    snapshot_time = EARLY_CLOSE_SNAPSHOT_TIME if is_early_close else NORMAL_SNAPSHOT_TIME
    return datetime.combine(quote_date, snapshot_time, tzinfo=EASTERN_TZ)


def settlement_policy_record(option_root: str, policy: SettlementConfig) -> dict[str, object]:
    if option_root != "SPX":
        raise ValueError(
            f"Unsupported option root for settlement policy: {option_root}. Only SPX is supported in v1."
        )
    return {
        "option_root": option_root,
        "settlement_style": policy.settlement_style,
        "proxy_time_eastern": policy.proxy_time_eastern.strftime("%H:%M"),
        "exact_clock": policy.exact_clock,
        "description": (
            "Standard SPX expiries settle to an A.M. SOQ/SET session. "
            "The configured proxy clock is an explicit approximation for ACT/365 timing, "
            "not an official exact settlement timestamp."
        ),
    }


def settlement_timestamp_eastern(
    expiration: date,
    option_root: str,
    policy: SettlementConfig,
) -> datetime:
    if option_root != "SPX":
        raise ValueError(
            f"Unsupported option root for settlement-time logic: {option_root}. "
            "Only SPX is supported in v1."
        )
    if policy.settlement_style != "AM_SOQ_PROXY":
        raise ValueError(
            f"Unsupported settlement style for {option_root}: {policy.settlement_style}."
        )
    return datetime.combine(expiration, policy.proxy_time_eastern, tzinfo=EASTERN_TZ)


def year_fraction_act365(start_ts: datetime, end_ts: datetime) -> float:
    if end_ts <= start_ts:
        raise ValueError(
            f"Settlement timestamp must be later than snapshot timestamp, got {start_ts} -> {end_ts}."
        )
    return (end_ts - start_ts).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)
