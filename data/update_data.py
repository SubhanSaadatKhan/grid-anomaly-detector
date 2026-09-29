"""Append the latest German grid load readings from ENTSO-E to data/grid_load.csv.

Runs every few hours from GitHub Actions (.github/workflows/fetch_data.yml).

The ENTSO-E API is flaky: it regularly answers with 503/599/404 during
maintenance, and sometimes has simply not published the latest hours yet
(NoMatchingDataError). None of that should crash the job, so this script:

  * fetches in one-day chunks, so one bad chunk doesn't lose the rest,
  * retries transient HTTP / network errors with backoff,
  * re-fetches a short look-back window every run, so hours that ENTSO-E
    publishes late (or that were missed during an outage) get filled in,
  * treats "no new data" and transient outages as warnings, not failures.

It only exits non-zero for problems a human must fix: a missing or rejected
API key, or data that has been stale for longer than STALE_AFTER.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

CSV_PATH = Path(__file__).resolve().parent / "grid_load.csv"
ZONE = "DE_LU"                                   # Germany/Luxembourg bidding zone
TZ = "Europe/Berlin"
CHUNK = pd.Timedelta(days=1)
LOOKBACK = pd.Timedelta(days=int(os.environ.get("BACKFILL_DAYS") or 3))
STALE_AFTER = pd.Timedelta(days=7)
RETRIES = 3
RETRY_DELAY = 10                                 # seconds, doubled after each retry
FATAL_STATUS = {400, 401, 403}                   # bad request / bad or revoked key


class FatalError(Exception):
    """An error that retrying will not fix."""


def warn(msg):
    # "::warning::" shows up as a yellow annotation on the GitHub Actions run.
    prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "WARNING: "
    print(f"{prefix}{msg}")


def fetch_chunk(client, start, end, retries=RETRIES, delay=RETRY_DELAY):
    """Return a DataFrame(timestamp, load_mw) for [start, end), or None if unavailable."""
    for attempt in range(1, retries + 1):
        try:
            ts = client.query_load(ZONE, start=start, end=end)
            break
        except NoMatchingDataError:
            return None                          # not published yet - try again next run
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in FATAL_STATUS:
                raise FatalError(f"ENTSO-E rejected the request (HTTP {status}). "
                                 "Check the ENTSOE_API_KEY secret.") from e
            err = f"HTTP {status}"
        except (requests.ConnectionError, requests.Timeout) as e:
            err = type(e).__name__
        if attempt == retries:
            warn(f"ENTSO-E unavailable for {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} "
                 f"({err} after {retries} attempts); will retry next run.")
            return None
        wait = delay * 2 ** (attempt - 1)
        print(f"  {err}, retrying in {wait}s (attempt {attempt}/{retries})")
        time.sleep(wait)

    if isinstance(ts, pd.DataFrame):
        ts = ts["Actual Load"] if "Actual Load" in ts.columns else ts.iloc[:, 0]
    df = ts.rename("load_mw").rename_axis("timestamp").reset_index()
    return df.dropna(subset=["load_mw"])


def fetch_range(client, start, end, **kw):
    """Fetch [start, end) in CHUNK-sized pieces. Returns (frames, n_failed_chunks)."""
    frames, failed = [], 0
    cursor = start
    while cursor < end:
        stop = min(cursor + CHUNK, end)
        print(f"Fetching {cursor:%Y-%m-%d %H:%M} -> {stop:%Y-%m-%d %H:%M}")
        df = fetch_chunk(client, cursor, stop, **kw)
        if df is None:
            failed += 1
        elif len(df):
            frames.append(df)
        cursor = stop
    return frames, failed


def load_existing(path=CSV_PATH):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def merge(existing, frames):
    new = pd.concat(frames) if frames else existing.iloc[0:0]
    new = new.assign(timestamp=pd.to_datetime(new["timestamp"], utc=True))
    return (pd.concat([existing, new])
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True))


def update(client, path=CSV_PATH, now=None, **kw):
    """Update the CSV in place. Returns the number of new rows."""
    existing = load_existing(path)
    last = existing["timestamp"].max()
    now = now or pd.Timestamp.now(tz=TZ)

    # Re-fetch a look-back window so late-published hours and outage gaps get filled.
    start = (last - LOOKBACK).tz_convert(TZ).floor("h")
    end = now.floor("15min")
    print(f"Last reading: {last}. Fetching {start} -> {end}")

    frames, failed = fetch_range(client, start, end, **kw)
    combined = merge(existing, frames)
    added = len(combined) - len(existing)

    if frames:
        # Always store UTC so every row uses the same offset (+00:00).
        combined.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S+00:00")
    print(f"Done: {added} new rows, {len(combined)} total, {failed} chunk(s) unavailable.")

    newest = combined["timestamp"].max()
    age = now - newest
    if age > STALE_AFTER:
        raise FatalError(f"Newest reading is {newest} ({age} old). ENTSO-E has been "
                         f"unreachable for over {STALE_AFTER.days} days - please investigate.")
    if added == 0:
        print("No new data this run (ENTSO-E may not have published it yet).")
    return added


def main():
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        print("::error::ENTSOE_API_KEY is not set. Add it under "
              "Settings -> Secrets and variables -> Actions.")
        return 1
    client = EntsoePandasClient(api_key=api_key, retry_count=3, retry_delay=10)
    try:
        update(client)
    except FatalError as e:
        print(f"::error::{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
