import sys
import os as os
import pandas as pd

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def date_to_calendar_week(date) -> tuple[int, int]:
    """Convert a date to its ISO (year, week) tuple, e.g. '2025-03-17' -> (2025, 12)."""
    iso = pd.Timestamp(date).isocalendar()
    return int(iso.year), int(iso.week)


def calendar_week_to_date(iso_year: int, iso_week: int, weekday: int = 1) -> pd.Timestamp:
    """Convert an ISO (year, week) back to a date. weekday is 1=Monday..7=Sunday (default: Monday)."""
    return pd.Timestamp.fromisocalendar(iso_year, iso_week, weekday)

def tHOY_to_date(tHOY: int, start_date: str = "2025-01-01") -> pd.Timestamp:
    """Convert a time index (tHOY) to a date, given a start date."""
    start_date = pd.Timestamp(start_date)
    return start_date + pd.Timedelta(hours=tHOY - 1)  

