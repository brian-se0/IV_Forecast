from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class EarlyCloseEntry:
    quote_date: date
    market_close_time_eastern: str
    source_name: str
    source_url: str
    notes: str


@dataclass(frozen=True)
class EarlyCloseCalendar:
    calendar_name: str
    coverage_start: date
    coverage_end: date
    entries: tuple[EarlyCloseEntry, ...]

    @property
    def dates(self) -> tuple[date, ...]:
        return tuple(entry.quote_date for entry in self.entries)

    def entries_in_range(self, start_date: date, end_date: date) -> list[EarlyCloseEntry]:
        _validate_manifest_coverage(self, start_date, end_date)
        return [entry for entry in self.entries if start_date <= entry.quote_date <= end_date]


def _validate_manifest_coverage(
    calendar: EarlyCloseCalendar,
    start_date: date,
    end_date: date,
) -> None:
    if start_date < calendar.coverage_start or end_date > calendar.coverage_end:
        raise ValueError(
            "The configured date range falls outside the checked-in early-close manifest coverage: "
            f"{start_date.isoformat()}..{end_date.isoformat()} versus "
            f"{calendar.coverage_start.isoformat()}..{calendar.coverage_end.isoformat()}."
        )


@lru_cache(maxsize=1)
def load_early_close_calendar() -> EarlyCloseCalendar:
    resource = files("ivs_forecast").joinpath("resources/cboe_us_options_early_closes.csv")
    entries: list[EarlyCloseEntry] = []
    with resource.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                EarlyCloseEntry(
                    quote_date=date.fromisoformat(row["quote_date"]),
                    market_close_time_eastern=row["market_close_time_eastern"],
                    source_name=row["source_name"],
                    source_url=row["source_url"],
                    notes=row["notes"],
                )
            )
    if not entries:
        raise ValueError("The checked-in early-close manifest is empty.")
    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.quote_date))
    quote_dates = [entry.quote_date for entry in ordered_entries]
    if len(set(quote_dates)) != len(quote_dates):
        raise ValueError("The checked-in early-close manifest contains duplicate quote_date rows.")
    return EarlyCloseCalendar(
        calendar_name="cboe_us_options_early_closes_manifest_v1",
        coverage_start=date(quote_dates[0].year, 1, 1),
        coverage_end=date(quote_dates[-1].year, 12, 31),
        entries=ordered_entries,
    )


def early_close_dates_in_range(start_date: date, end_date: date) -> list[date]:
    calendar = load_early_close_calendar()
    return [entry.quote_date for entry in calendar.entries_in_range(start_date, end_date)]


def early_close_date_set(start_date: date, end_date: date) -> set[date]:
    return set(early_close_dates_in_range(start_date, end_date))
