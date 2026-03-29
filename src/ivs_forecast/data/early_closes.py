from __future__ import annotations

from datetime import date, timedelta


def _thanksgiving(year: int) -> date:
    november_first = date(year, 11, 1)
    first_thursday_offset = (3 - november_first.weekday()) % 7
    first_thursday = november_first + timedelta(days=first_thursday_offset)
    return first_thursday + timedelta(weeks=3)


def _previous_business_day(day: date) -> date:
    candidate = day - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _curated_year_dates(year: int) -> set[date]:
    dates = {_thanksgiving(year) + timedelta(days=1)}
    july_fourth = date(year, 7, 4)
    if july_fourth.weekday() in {0, 6}:
        dates.add(_previous_business_day(july_fourth))
    elif july_fourth.weekday() not in {5}:
        dates.add(_previous_business_day(july_fourth))
    christmas = date(year, 12, 25)
    if christmas.weekday() not in {5}:
        candidate = _previous_business_day(christmas)
        observed_holiday = date(year, 12, 24) if christmas.weekday() == 5 else None
        if candidate != observed_holiday:
            dates.add(candidate)
    return dates


CURATED_EARLY_CLOSE_DATES: tuple[date, ...] = tuple(
    sorted(day for year in range(2004, 2027) for day in _curated_year_dates(year))
)


def early_close_dates_in_range(start_date: date, end_date: date) -> list[date]:
    return [day for day in CURATED_EARLY_CLOSE_DATES if start_date <= day <= end_date]
