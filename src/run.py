from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import parser as dtparse
from icalendar import Calendar, Event

# ----------------------------
# Data models / config
# ----------------------------


@dataclass
class DivisionConfig:
    slug: str
    name: str
    tz: str
    default_duration_min: int
    links: Dict[str, str]
    include_teams: Optional[List[str]] = None


@dataclass
class TelegramChannelConfig:
    name: str
    chat_id: str
    division_slug: str
    fixtures_team_filter: Optional[str] = None
    results_scope: str = "league"  # league | all | none
    enabled: bool = True


@dataclass
class Match:
    kickoff_local: datetime
    home: str
    away: str
    competition: str
    result_home: Optional[int]
    result_away: Optional[int]
    status: str  # Scheduled / Played / Postponed / Home Walkover / Away Walkover
    source: str
    match_type: str = ""  # If we ever want to tag leagues vs cups (e.g. "L")
    raw_status: str = ""  # Raw text from site, e.g. "Postponed"
    score_text: str = ""  # e.g. "2 - 7 (HT 1-2)" or "Postponed P - P"
    fixture_id: str = ""  # If present in HTML (data-fixtureid)


# ----------------------------
# Helpers
# ----------------------------


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dirs() -> None:
    Path("output").mkdir(exist_ok=True)
    Path("snapshots").mkdir(exist_ok=True)
    Path("status").mkdir(exist_ok=True)


def http_get(url: str) -> requests.Response:
    headers = {
        "User-Agent": "football-fixtures-ics/1.0 (+https://github.com/JustGeary/football-fixtures-ics)"
    }
    return requests.get(url, headers=headers, timeout=30)


def safe_dtparse(text: str, tz: ZoneInfo) -> Optional[datetime]:
    text = norm_ws(text)
    if not text:
        return None
    try:
        dt = dtparse.parse(text, dayfirst=True)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def normalise_team_name(name: str) -> str:
    n = norm_ws(name)
    n = re.sub(r"\s*\[[^\]]+\]\s*$", "", n)  # remove trailing [Dorset]
    n = re.sub(r"\s+", " ", n)
    return n


_DATE_TIME_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\b")


def detect_status_cell(text: str) -> Optional[str]:
    t = norm_ws(text).lower()
    if t == "postponed":
        return "Postponed"
    if t == "home walkover":
        return "Home Walkover"
    if t == "away walkover":
        return "Away Walkover"
    return None


def parse_score_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts "2 - 7" style scores from the main text.
    Returns (home, away) or (None, None).
    """
    text = norm_ws(text)
    if not text:
        return None, None
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None, None


# ----------------------------
# Parsing fixtures / results
# ----------------------------


def parse_fixtures_table(html: str, div: DivisionConfig) -> List[Match]:
    soup = BeautifulSoup(html, "html.parser")
    tz = ZoneInfo(div.tz)

    matches: List[Match] = []

    # Attempt to derive a default competition name from page
    div_name_el = soup.select_one(".contentheading, h1, h2")
    division_name = norm_ws(div_name_el.get_text(" ", strip=True)) if div_name_el else div.name

    for row in soup.select("table.table-fixtures tbody tr"):
        tds = row.find_all("td")
        if not tds:
            continue

        # Date/time – look for anything that matches date+time pattern or is in a dedicated KO col
        dt_text = ""
        ko_cell = row.select_one(".ko-datetime-col, .ko-time-col, .ko-date-col")
        if ko_cell:
            dt_text = norm_ws(ko_cell.get_text(" ", strip=True))
        else:
            for td in tds:
                m = _DATE_TIME_RE.search(norm_ws(td.get_text(" ", strip=True)))
                if m:
                    dt_text = m.group(0)
                    break

        dt = safe_dtparse(dt_text, tz)
        if dt is None:
            continue

        home_el = row.select_one(".home-team-col .team-name a, .home-team-col a, .home-team-col")
        away_el = row.select_one(".road-team-col .team-name a, .road-team-col a, .road-team-col")
        home = normalise_team_name(home_el.get_text(" ", strip=True) if home_el else "")
        away = normalise_team_name(away_el.get_text(" ", strip=True) if away_el else "")

        if not home or not away:
            continue

        comp_el = row.select_one(".fg-col p, .fg-col")
        comp = norm_ws(comp_el.get_text(" ", strip=True)) if comp_el else division_name

        # Fixtures page usually doesn't have scores, but may have status text
        status = "Scheduled"
        raw_status = ""
        score_text = ""
        result_home: Optional[int] = None
        result_away: Optional[int] = None

        status_el = row.select_one(".status-col, .score-col")
        if status_el:
            st_text = norm_ws(status_el.get_text(" ", strip=True))
            detected = detect_status_cell(st_text)
            if detected:
                status = detected
                raw_status = detected
                score_text = st_text

        fixture_id = row.get("data-fixtureid", "")

        matches.append(
            Match(
                kickoff_local=dt,
                home=home,
                away=away,
                competition=comp,
                result_home=result_home,
                result_away=result_away,
                status=status,
                source="fixtures",
                match_type="",  # could infer league/cup later
                raw_status=raw_status,
                score_text=score_text,
                fixture_id=fixture_id,
            )
        )

    return matches


def parse_results_table(html: str, div: DivisionConfig) -> List[Match]:
    soup = BeautifulSoup(html, "html.parser")
    tz = ZoneInfo(div.tz)

    matches: List[Match] = []

    div_name_el = soup.select_one(".contentheading, h1, h2")
    division_name = norm_ws(div_name_el.get_text(" ", strip=True)) if div_name_el else div.name

    for row in soup.select("table.table-results tbody tr"):
        tds = row.find_all("td")
        if not tds:
            continue

        dt_text = ""
        ko_cell = row.select_one(".ko-datetime-col, .ko-time-col, .ko-date-col")
        if ko_cell:
            dt_text = norm_ws(ko_cell.get_text(" ", strip=True))
        else:
            for td in tds:
                m = _DATE_TIME_RE.search(norm_ws(td.get_text(" ", strip=True)))
                if m:
                    dt_text = m.group(0)
                    break

        dt = safe_dtparse(dt_text, tz)
        if dt is None:
            continue

        home_el = row.select_one(".home-team-col .team-name a, .home-team-col a, .home-team-col")
        away_el = row.select_one(".road-team-col .team-name a, .road-team-col a, .road-team-col")
        home = normalise_team_name(home_el.get_text(" ", strip=True) if home_el else "")
        away = normalise_team_name(away_el.get_text(" ", strip=True) if away_el else "")

        if not home or not away:
            continue

        comp_el = row.select_one(".fg-col p, .fg-col")
        comp = norm_ws(comp_el.get_text(" ", strip=True)) if comp_el else division_name

        status = "Played"
        raw_status = ""
        score_text = ""
        result_home: Optional[int] = None
        result_away: Optional[int] = None

        score_el = row.select_one(".score-col, .status-col")
        if score_el:
            st_text = norm_ws(score_el.get_text(" ", strip=True))
            score_text = st_text

            detected = detect_status_cell(st_text)
            if detected:
                status = detected
                raw_status = detected
            else:
                rh, ra = parse_score_text(st_text)
                result_home, result_away = rh, ra

        fixture_id = row.get("data-fixtureid", "")

        matches.append(
            Match(
                kickoff_local=dt,
                home=home,
                away=away,
                competition=comp,
                result_home=result_home,
                result_away=result_away,
                status=status,
                source="results",
                match_type="",
                raw_status=raw_status,
                score_text=score_text,
                fixture_id=fixture_id,
            )
        )

    return matches


# ----------------------------
# Merge fixtures/results per division
# ----------------------------


def merge_matches(fixtures: List[Match], results: List[Match]) -> List[Match]:
    matches_by_key: Dict[str, Match] = {}

    def key_for(m: Match) -> str:
        return f"{m.fixture_id or ''}|{m.kickoff_local.isoformat()}|{m.home}|{m.away}|{m.competition}"

    for m in fixtures:
        matches_by_key[key_for(m)] = m

    for r in results:
        k = key_for(r)
        if k in matches_by_key:
            existing = matches_by_key[k]
            # Update with result info
            existing.result_home = r.result_home
            existing.result_away = r.result_away
            existing.status = r.status
            existing.raw_status = r.raw_status
            existing.score_text = r.score_text
            existing.source = "fixtures+results"
        else:
            matches_by_key[k] = r

    return list(matches_by_key.values())


# ----------------------------
# Calendar / ICS
# ----------------------------


def event_uid(team_name: str, m: Match) -> str:
    """
    Stable-ish UID so re-runs update instead of duplicating.
    """
    base = (
        f"{team_name}|"
        f"{m.kickoff_local.replace(second=0, microsecond=0).isoformat()}|"
        f"{m.home}|{m.away}|{m.competition}"
    )
    sha = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return f"{sha}@football-fixtures-ics"


def build_team_calendar_bytes(team_name: str, matches: List[Match], div: DivisionConfig) -> bytes:
    tz = ZoneInfo(div.tz)
    cal = Calendar()
    cal.add("prodid", "-//football-fixtures-ics//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("method", "PUBLISH")
    cal.add("x-wr-calname", f"{team_name} Fixtures")
    cal.add("x-wr-timezone", div.tz)

    # include past + future window
    start = datetime.now(tz).date() - timedelta(days=365)
    end = datetime.now(tz).date() + timedelta(days=365)

    team_norm = normalise_team_name(team_name).lower()

    for m in matches:
        if not (start <= m.kickoff_local.date() <= end):
            continue

        e = Event()
        e.add("uid", event_uid(team_name, m))
        e.add("dtstamp", now_utc())

        # iOS-friendly: UTC DTSTART/DTEND
        dtstart_utc = m.kickoff_local.astimezone(timezone.utc)
        dtend_utc = dtstart_utc + timedelta(minutes=div.default_duration_min)
        e.add("dtstart", dtstart_utc)
        e.add("dtend", dtend_utc)

        # Home/Away + competition + result in SUMMARY
        is_home = normalise_team_name(m.home).lower() == team_norm
        opponent = m.away if is_home else m.home
        venue = "Home" if is_home else "Away"

        comp = (m.competition or "").strip()
        comp_lower = comp.lower()
        mt = (m.match_type or "").upper()
        if mt == "L":
            comp_label = "League"
        elif "cup" in comp_lower and "final" in comp_lower:
            comp_label = "Cup Final"
        elif "cup" in comp_lower:
            comp_label = "Cup"
        elif "friendly" in comp_lower:
            comp_label = "Friendly"
        elif "division" in comp_lower or "league" in comp_lower:
            comp_label = "League"
        elif comp:
            comp_label = comp
        else:
            comp_label = "Fixture"

        summary = f"{venue} vs {opponent} ({comp_label})".strip()

        if m.status in {"Postponed", "Home Walkover", "Away Walkover"}:
            summary = f"[{m.status}] {summary}"

        if m.result_home is not None and m.result_away is not None:
            summary = f"{summary} — {m.result_home}–{m.result_away}"

        e.add("summary", summary)

        desc: List[str] = []
        if m.competition:
            desc.append(f"Competition: {m.competition}")
        if m.match_type:
            desc.append(f"Type: {m.match_type}")

        if m.result_home is not None and m.result_away is not None:
            desc.append(f"Result: {m.home} {m.result_home}–{m.result_away} {m.away}")
            if m.score_text and "(" in m.score_text:
                # e.g. "(HT 1-2)"
                desc.append(f"Notes: {m.score_text[m.score_text.find('('):]}")
        elif m.score_text:
            desc.append(f"Status/Score: {m.score_text}")
        elif m.raw_status:
            desc.append(f"Status: {m.raw_status}")

        if div.links.get("table"):
            desc.append(f"Table: {div.links['table']}")
        if div.links.get("fixtures"):
            desc.append(f"Fixtures: {div.links['fixtures']}")
        if div.links.get("results"):
            desc.append(f"Results: {div.links['results']}")

        if desc:
            e.add("description", "\n".join(desc))

        if div.links.get("fixtures"):
            e.add("url", div.links["fixtures"])

        cal.add_component(e)

    return cal.to_ical()


# ----------------------------
# Snapshotting for diffs / Telegram
# ----------------------------


def matches_to_snapshot_list(matches: List[Match]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in matches:
        out.append(
            {
                "kickoff_local": m.kickoff_local.isoformat(),
                "home": m.home,
                "away": m.away,
                "competition": m.competition,
                "result_home": m.result_home,
                "result_away": m.result_away,
                "status": m.status,
                "score_text": m.score_text,
                "fixture_id": m.fixture_id,
                "match_type": m.match_type,
            }
        )
    return out


def load_snapshot_list(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_snapshot_list(path: Path, data: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _snap_key(d: Dict[str, Any]) -> str:
    return f"{d.get('kickoff_local')}|{d.get('home')}|{d.get('away')}|{d.get('competition')}"


# ----------------------------
# Telegram helpers
# ----------------------------


def telegram_send(text: str, chat_id: Optional[str] = None) -> None:
    """
    Uses TELEGRAM_BOT_TOKEN env var.
    If chat_id not supplied, falls back to TELEGRAM_CHAT_ID env var (single-channel mode).
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return

    cid = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if not cid:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": cid, "text": text}, timeout=15)
    except Exception:
        # Don't crash the whole run on Telegram errors
        return


def _involves_team(d: Dict[str, Any], team_filter: Optional[str]) -> bool:
    if not team_filter:
        return True
    tf = normalise_team_name(team_filter).lower()
    return (
        normalise_team_name(d.get("home", "")).lower() == tf
        or normalise_team_name(d.get("away", "")).lower() == tf
    )


def _format_ko(iso_str: str, tz_name: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    dt = dt.astimezone(tz)
    return dt.strftime("%a %d %b, %H:%M")  # e.g. "Sun 18 Jan, 14:00"


def _comp_label_from_snap(d: Dict[str, Any]) -> str:
    mt = (d.get("match_type") or "").upper()
    if mt == "L":
        return "League"

    comp = (d.get("competition") or "").strip()
    if not comp:
        return "Fixture"

    r = comp.lower()
    if "friendly" in r:
        return "Friendly"
    if "cup" in r and "final" in r:
        return "Cup Final"
    if "cup" in r:
        return "Cup"
    if "division" in r or "league" in r:
        return "League"
    return comp


def _fixture_label(d: Dict[str, Any], fixtures_team_filter: Optional[str]) -> str:
    home = d.get("home", "")
    away = d.get("away", "")
    comp_label = _comp_label_from_snap(d)

    if fixtures_team_filter:
        tf = normalise_team_name(fixtures_team_filter).lower()
        home_norm = normalise_team_name(home).lower()
        is_home = home_norm == tf
        venue = "Home" if is_home else "Away"
        opponent = away if is_home else home
        return f"{venue} vs {opponent} ({comp_label})"

    return f"{home} vs {away} ({comp_label})"


def _fixture_line(d: Dict[str, Any], fixtures_team_filter: Optional[str], tz_name: str) -> str:
    ko = _format_ko(d.get("kickoff_local") or "", tz_name)
    return f"{ko} – {_fixture_label(d, fixtures_team_filter)}"


def _result_line(d: Dict[str, Any], tz_name: str) -> Optional[str]:
    rh = d.get("result_home")
    ra = d.get("result_away")
    if rh is None or ra is None:
        return None

    ko = _format_ko(d.get("kickoff_local") or "", tz_name)
    comp_label = _comp_label_from_snap(d)

    base = f"{ko} – {d.get('home', '')} {rh}–{ra} {d.get('away', '')}"
    if comp_label != "League":
        base += f" ({comp_label})"
    return base


def build_change_messages(
    div_name: str,
    prev_snap: Optional[List[Dict[str, Any]]],
    curr_snap: List[Dict[str, Any]],
    *,
    fixtures_team_filter: Optional[str] = None,  # applies to fixture changes only
    results_scope: str = "league",               # league | all | none
    tz_name: str = "Europe/London",
    max_lines: int = 14,
) -> List[str]:
    """
    Hybrid alert rules:
      - Fixture changes (new/removed/KO/status): filtered by fixtures_team_filter if set.
      - Results: posted division-wide depending on results_scope (league/all/none),
        but each time a result changes we send the cumulative list of today's results.
    """
    prev_snap = prev_snap or []
    prev_map = {_snap_key(d): d for d in prev_snap}
    curr_map = {_snap_key(d): d for d in curr_snap}

    fixture_lines: List[str] = []
    result_lines: List[str] = []

    def is_league(d: Dict[str, Any]) -> bool:
        return _comp_label_from_snap(d) == "League"

    def include_result(d: Dict[str, Any]) -> bool:
        if results_scope == "none":
            return False
        if results_scope == "all":
            return True
        return is_league(d)

    # Track whether any result changed this run
    results_triggered = False

    # Removed fixtures (fixture-change alerts only)
    for k, p in prev_map.items():
        if k not in curr_map and _involves_team(p, fixtures_team_filter):
            fixture_lines.append(
                f"➖ Removed: {_fixture_line(p, fixtures_team_filter, tz_name)}"
            )

    # Added/changed fixtures + detect result changes
    for k, c in curr_map.items():
        p = prev_map.get(k)

        # New fixture (fixture-change alerts only)
        if p is None:
            if _involves_team(c, fixtures_team_filter):
                fixture_lines.append(
                    f"➕ New: {_fixture_line(c, fixtures_team_filter, tz_name)}"
                )
            continue

        # KO / Status changes (fixture-change alerts only)
        if _involves_team(c, fixtures_team_filter):
            p_ko = p.get("kickoff_local")
            c_ko = c.get("kickoff_local")
            if p_ko != c_ko:
                label = _fixture_label(c, fixtures_team_filter)
                p_ko_str = _format_ko(p_ko or "", tz_name)
                c_ko_str = _format_ko(c_ko or "", tz_name)
                fixture_lines.append(
                    f"⏱️ KO change: {label} {p_ko_str} → {c_ko_str}"
                )

            p_status = p.get("status")
            c_status = c.get("status")
            if p_status != c_status and c_status:
                label = _fixture_label(c, fixtures_team_filter)
                fixture_lines.append(
                    f"⛔ Status: {label} → {c_status}"
                )

        # Results (division-wide, scope-controlled) – detect any changes
        if include_result(c):
            p_rh, p_ra = p.get("result_home"), p.get("result_away")
            c_rh, c_ra = c.get("result_home"), c.get("result_away")

            prev_has = (p_rh is not None and p_ra is not None)
            curr_has = (c_rh is not None and c_ra is not None)

            # Any new or changed result triggers a cumulative "today's results" list
            if (not prev_has and curr_has) or (
                prev_has and curr_has and (p_rh != c_rh or p_ra != c_ra)
            ):
                results_triggered = True

    # Build today's cumulative results list if any result changed
    results_heading = "*Results*"
    if results_triggered:
        today_local = datetime.now(ZoneInfo(tz_name)).date()
        all_results_today: List[Dict[str, Any]] = []

        for d in curr_map.values():
            if not include_result(d):
                continue

            rh, ra = d.get("result_home"), d.get("result_away")
            if rh is None or ra is None:
                continue

            ko_str = d.get("kickoff_local") or ""
            try:
                ko_dt = datetime.fromisoformat(ko_str)
                if ko_dt.tzinfo is None:
                    ko_dt = ko_dt.replace(tzinfo=timezone.utc)
                ko_date_local = ko_dt.astimezone(ZoneInfo(tz_name)).date()
            except Exception:
                ko_date_local = today_local

            if ko_date_local != today_local:
                continue

            all_results_today.append(d)

        # sort by kickoff time
        all_results_today.sort(key=lambda d: d.get("kickoff_local") or "")

        for d in all_results_today:
            line = _result_line(d, tz_name)
            if line:
                result_lines.append(line)

        if all_results_today:
            first_ko_str = all_results_today[0].get("kickoff_local") or ""
            try:
                dt = datetime.fromisoformat(first_ko_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                tz = ZoneInfo(tz_name)
                dt_local = dt.astimezone(tz)
                results_heading = f"*Results ({dt_local.strftime('%a %d %b')})*"
            except Exception:
                pass

    if not fixture_lines and not result_lines:
        return []

    lines: List[str] = []

    if fixture_lines:
        lines.append("*Fixture changes*")
        lines.extend(fixture_lines)

    if result_lines:
        if lines:
            lines.append("")  # spacing between sections
        lines.append(results_heading)
        lines.extend(result_lines)

    header = f"📣 {div_name} updates ({now_utc().strftime('%Y-%m-%d %H:%MZ')})"

    # Chunking so we don't exceed comfortable message size
    chunks: List[List[str]] = []
    buf: List[str] = [header]
    for ln in lines[:max_lines]:
        if len(buf) >= 1 + 18:
            chunks.append(buf)
            buf = [header + " (cont.)"]
        buf.append(ln)
    chunks.append(buf)

    if len(lines) > max_lines:
        chunks[-1].append(f"…and {len(lines) - max_lines} more changes.")

    return ["\n".join(c) for c in chunks]


# ----------------------------
# Config loading
# ----------------------------


def load_divisions() -> List[DivisionConfig]:
    divs: List[DivisionConfig] = []
    for p in sorted(Path("divisions").glob("*.yml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            raise RuntimeError(f"Division config is not a mapping in {p}")

        links = raw.get("links")
        if not isinstance(links, dict):
            raise RuntimeError(f"Division config missing/invalid links in {p}")

        for k in ("fixtures", "results"):
            if not links.get(k):
                raise RuntimeError(f"Division config links.{k} missing/blank in {p}")

        divs.append(
            DivisionConfig(
                slug=str(raw["slug"]).strip(),
                name=str(raw["name"]).strip(),
                tz=str(raw.get("timezone", "Europe/London")).strip(),
                default_duration_min=int(raw.get("default_duration_min", 120)),
                links=links,
                include_teams=raw.get("include_teams"),
            )
        )

    if not divs:
        raise RuntimeError("No divisions/*.yml found")

    return divs


def load_telegram_channels(path: Path = Path("telegram_channels.yml")) -> List[TelegramChannelConfig]:
    """
    telegram_channels.yml example:

    channels:
      - name: poole-nonadmin
        chat_id: -1002984118908
        division_slug: u18-division-1
        fixtures_team_filter: "Poole Town FC Wessex U18 Colts"
        results_scope: league
        enabled: true
    """
    if not path.exists():
        return []

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    channels: List[TelegramChannelConfig] = []

    for c in raw.get("channels", []):
        channels.append(
            TelegramChannelConfig(
                name=str(c.get("name", "")).strip(),
                chat_id=str(c.get("chat_id", "")).strip(),
                division_slug=str(c.get("division_slug", "")).strip(),
                fixtures_team_filter=(str(c.get("fixtures_team_filter", "")).strip() or None),
                results_scope=str(c.get("results_scope", "league")).strip().lower(),
                enabled=bool(c.get("enabled", True)),
            )
        )

    return channels


def guess_pages_base() -> str:
    """
    Try to guess the GitHub Pages base URL from env vars.
    """
    env_base = os.getenv("PAGES_BASE_URL", "").strip()
    if env_base:
        return env_base.rstrip("/")

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if repo and "/" in repo:
        owner, repo_name = repo.split("/", 1)
        return f"https://{owner}.github.io/{repo_name}"

    return "https://justgeary.github.io/football-fixtures-ics"


def slugify_team_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "team"


def build_index_html(teams: List[str], repo_pages_base: str, div: DivisionConfig) -> str:
    lines = [
        "<html>",
        "<head><title>Football Fixtures ICS</title></head>",
        "<body>",
        f"<h1>{div.name} – Fixtures Calendars</h1>",
        "<ul>",
    ]
    for team in sorted(teams):
        ics_name = slugify_team_name(team) + ".ics"
        url = f"{repo_pages_base}/{ics_name}"
        lines.append(f'<li><a href="{url}">{team}</a></li>')
    lines.extend(["</ul>", "</body>", "</html>"])
    return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ensure_dirs()
    divisions = load_divisions()
    pages_base = guess_pages_base()

    # Multi-channel config (optional). If file missing, fallback to single channel env
    channels = load_telegram_channels()

    summary: Dict[str, Any] = {"divisions": []}

    for div in divisions:
        tz = ZoneInfo(div.tz)
        division_name = div.name

        fixtures_url = div.links["fixtures"]
        results_url = div.links["results"]

        fx_resp = http_get(fixtures_url)
        rs_resp = http_get(results_url)

        if fx_resp.status_code != 200 or rs_resp.status_code != 200:
            msg = f"⚠️ HTTP error ({div.name}): fixtures {fx_resp.status_code}, results {rs_resp.status_code}"
            # Send to all relevant channels (or fallback env)
            relevant = [c for c in channels if c.division_slug == div.slug and c.enabled]
            if relevant:
                for ch in relevant:
                    telegram_send(msg, ch.chat_id)
            else:
                telegram_send(msg)
            continue

        fixtures = parse_fixtures_table(fx_resp.text, div)
        results = parse_results_table(rs_resp.text, div)
        merged = merge_matches(fixtures, results)

        # Save ICS per team if include_teams specified
        teams: List[str] = []
        if div.include_teams:
            # Just list of names to filter by
            for team in div.include_teams:
                team_norm = normalise_team_name(team).lower()
                team_matches = [
                    m
                    for m in merged
                    if normalise_team_name(m.home).lower() == team_norm
                    or normalise_team_name(m.away).lower() == team_norm
                ]
                if not team_matches:
                    continue

                teams.append(team)
                ics_bytes = build_team_calendar_bytes(team, team_matches, div)
                ics_name = slugify_team_name(team) + ".ics"
                Path("output", ics_name).write_bytes(ics_bytes)

        # Division index.html for convenience (if teams exist)
        if teams:
            index_html = build_index_html(teams, pages_base, div)
            Path("output", f"{div.slug}-index.html").write_text(index_html, encoding="utf-8")

        # Snapshot + Telegram diffs
        snapshot_path = Path("snapshots") / f"{div.slug}.json"
        prev_latest = load_snapshot_list(snapshot_path)
        curr_latest = matches_to_snapshot_list(merged)
        save_snapshot_list(snapshot_path, curr_latest)

        changed = prev_latest is None or prev_latest != curr_latest

        # Telegram logic
        if changed and os.getenv("TELEGRAM_BOT_TOKEN"):
            if channels:
                for ch in channels:
                    if ch.division_slug != div.slug:
                        continue
                    for m in build_change_messages(
                        div.name,
                        prev_latest,
                        curr_latest,
                        fixtures_team_filter=ch.fixtures_team_filter,
                        results_scope=ch.results_scope,
                        tz_name=div.tz,
                    ):
                        telegram_send(m, ch.chat_id)
            else:
                # Single-channel env var mode
                fixtures_team_filter = os.getenv("TELEGRAM_TEAM_FILTER", "").strip() or None
                results_scope = os.getenv("TELEGRAM_RESULTS_SCOPE", "league").strip().lower()
                for m in build_change_messages(
                    div.name,
                    prev_latest,
                    curr_latest,
                    fixtures_team_filter=fixtures_team_filter,
                    results_scope=results_scope,
                    tz_name=div.tz,
                ):
                    telegram_send(m)

        summary["divisions"].append(
            {
                "slug": div.slug,
                "name": division_name,
                "timezone": div.tz,
                "published_team_count": len(teams),
                "published_teams": teams,
                "results_parsed": len(results),
                "fixtures_parsed": len(fixtures),
                "snapshot_changed": changed,
            }
        )

    Path("status", "last_run.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now_utc().isoformat(),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
