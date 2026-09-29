"""Tests for data/update_data.py using a fake ENTSO-E client (no API key needed).

Each failure mode here was seen in real GitHub Actions runs before the fix.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests
from entsoe.exceptions import NoMatchingDataError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import update_data  # noqa: E402

NOW = pd.Timestamp("2026-09-21 04:00", tz="Europe/Berlin")


def http_error(status):
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status} error", response=resp)


class FakeClient:
    """Serves 15-minute readings; `errors` is a list consumed one per call."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = 0

    def query_load(self, zone, start, end):
        self.calls += 1
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err
        idx = pd.date_range(start, end, freq="15min", inclusive="left")
        return pd.DataFrame({"Actual Load": 50000.0}, index=idx)


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "grid_load.csv"
    ts = pd.date_range("2026-09-19 00:00", "2026-09-19 21:45", freq="15min", tz="UTC")
    pd.DataFrame({"timestamp": ts, "load_mw": 40000.0}).to_csv(path, index=False)
    return path


def run(client, csv, now=NOW):
    return update_data.update(client, path=csv, now=now, delay=0)


def test_appends_new_rows_in_utc(csv):
    added = run(FakeClient(), csv)
    assert added > 0
    text = csv.read_text()
    assert "+02:00" not in text                      # no mixed offsets
    df = update_data.load_existing(csv)
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()


def test_no_matching_data_is_not_a_failure(csv):
    # Runs 416/417: ENTSO-E had not published new data yet.
    before = csv.read_text()
    client = FakeClient(errors=[NoMatchingDataError()] * 10)
    assert run(client, csv) == 0
    assert csv.read_text() == before


@pytest.mark.parametrize("status", [503, 599, 404, 429])
def test_transient_http_errors_are_retried(csv, status):
    # Runs 331-343 (503), 363-367 (404), 370 (599).
    client = FakeClient(errors=[http_error(status), http_error(status)])
    assert run(client, csv) > 0


def test_persistent_outage_warns_but_keeps_partial_data(csv, capsys):
    client = FakeClient(errors=[None] + [http_error(503)] * 100)
    added = run(client, csv)
    assert added > 0                                # first chunk was saved
    assert "unavailable" in capsys.readouterr().out


def test_bad_api_key_fails_loudly(csv):
    with pytest.raises(update_data.FatalError, match="ENTSOE_API_KEY"):
        run(FakeClient(errors=[http_error(401)]), csv)


def test_stale_data_fails_loudly(csv):
    client = FakeClient(errors=[http_error(503)] * 1000)
    with pytest.raises(update_data.FatalError, match="stale|unreachable"):
        run(client, csv, now=NOW + pd.Timedelta(days=10))


def test_lookback_fills_gaps(csv):
    df = update_data.load_existing(csv)
    df = df[(df["timestamp"].dt.hour < 5) | (df["timestamp"].dt.hour > 10)]  # punch a hole
    df.to_csv(csv, index=False)
    run(FakeClient(), csv)
    filled = update_data.load_existing(csv)
    assert filled["timestamp"].diff().max() == pd.Timedelta("15min")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    assert update_data.main() == 1
