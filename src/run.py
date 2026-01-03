from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from icalendar import Calendar, Event


# ----------------------------
# Config + Models
# ----------------------------

@dataclass(frozen=True)
class DivisionConfig:
    slug: str
    name: str
    tz: str
    default_duration_min: int
    links: Dict[str, str]
    include_teams: Optional[List[str]] = None


@dataclass
class Match:
    kickoff_local: datetime
    home: str
    away: str
    competition: str
    result_home: Optional[int] = None
    result_away: Optional[int] = None
    status: str = "Scheduled"  # Scheduled / Played / Postponed / Cancelled / TBC
    source: str = ""           # "fixtures" or "results"


# ----------------------------
# Helpers
# ----------------------------

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def http_get(url: str, *, timeout: int = 45) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; football-fixtures-ics/1.0; +https://github.com/JustGeary/football-fixtures-ics)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Connection": "keep-alive",
    }
    # Simple retry with backoff + jitter
    last_exc: Optional[Exception] = None
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            # Retry 429/5xx
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** attempt) + (0.2 * attempt))
                continue
            return resp
        except Exception as e:
            last_exc = e
            time.sleep((2 ** attempt) + 0.3)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def looks_like_block_page(html: str) -> bool:
    low = html.lower()
    # very rough heuristics
    return ("cloudflare" in low and "attention required" in low) or ("access denied" in low) or ("forbidden" in low and "html" in low)


def ensure_dirs() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)


def load_divisions() -> List[DivisionConfig]:
    divs: List[DivisionConfig] = []
    for p in sorted(Path("divisions").glob("*.yml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        divs.append(
            DivisionConfig(
                slug=raw["slug"],
                name=raw["name"],
                tz=raw.get("timezone", "Europe/London"),
                default_duration_min=int(raw.get("default_duration_min", 120)),
                links=raw["links"],
                include_teams=raw.get("include_teams"),
            )
        )
    if not divs:
        raise RuntimeError("No divisions/*.yml found")
    return divs


# ----------------------------
# Parsing
# ----------------------------

def _find_best_table(soup: BeautifulSoup) -> Optional[Any]:
    # Full-Time pages commonly contain one primary results/fixtures table.
    # We'll pick the first table that has headers like Home/Away OR contains many team-like cells.
    tables = soup.find_all("table")
    if not tables:
        return None

    def score_table(tbl: Any) -> int:
        score = 0
        th_text = " ".join([th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")])
        if "home" in th_text: score += 3
        if "away" in th_text: score += 3
        if "date" in th_text: score += 2
        if "time" in th_text: score += 2
        # crude: more rows => likely the main table
        score += min(len(tbl.find_all("tr")), 200) // 10
        return score

    best = max(tables, key=score_table)
    if score_table(best) < 5:
        return None
    return best


def _parse_datetime(text: str, tz: ZoneInfo) -> Optional[datetime]:
    text = text.strip()
    if not text:
        return None
    try:
        dt = dtparser.parse(text, dayfirst=True, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)
        return dt
    except Exception:
        return None


def parse_fixtures(html: str, tz: ZoneInfo) -> List[Match]:
    soup = BeautifulSoup(html, "lxml")
    tbl = _find_best_table(soup)
    if tbl is None:
        return []

    matches: List[Match] = []
    headers = [th.get_text(" ", strip=True) for th in tbl.find_all("th")]
    header_row = " | ".join(headers).lower()

    # Generic row parsing: try to extract date/time + home/away from cells.
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        cells = [td.get_text(" ", strip=True) for td in tds]
        joined = " ".join(cells).strip()
        if not joined:
            continue

        # Heuristic: find home/away by looking for " v " or a dedicated score column
        # Common patterns:
        #  - [date, time, home, away, venue/comp]
        date_candidate = None
        time_candidate = None

        # try first 2 columns for date/time-like strings
        if len(cells) >= 2:
            date_candidate = cells[0]
            time_candidate = cells[1]
        dt = _parse_datetime(f"{date_candidate} {time_candidate}", tz) if date_candidate else None

        # fallback: try anywhere for a date-like token
        if dt is None:
            dt = _parse_datetime(joined, tz)

        # Find two team-ish cells: longest alpha strings excluding obvious non-team columns
        teamish = [c for c in cells if len(c) >= 3 and not re.search(r"\b\d{1,2}:\d{2}\b", c)]
        # remove date-looking entries
        teamish = [c for c in teamish if not re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", c)]
        # remove score-like
        teamish = [c for c in teamish if not re.fullmatch(r"\d+\s*-\s*\d+", c)]

        if len(teamish) < 2 or dt is None:
            continue

        # choose home/away as first two distinct teamish values
        home = teamish[0].strip()
        away = next((x.strip() for x in teamish[1:] if x.strip() != home), "").strip()
        if not away:
            continue

        # competition/round: best-effort from remaining cells
        comp = ""
        # if headers mention Competition, use last cell as comp guess
        if "competition" in header_row or "round" in header_row:
            comp = teamish[-1] if teamish[-1] not in (home, away) else ""
        if not comp:
            # fallback: use page title-ish marker or blank
            comp = ""

        matches.append(
            Match(
                kickoff_local=dt,
                home=home,
                away=away,
                competition=comp,
                source="fixtures",
            )
        )
    return matches


def parse_results(html: str, tz: ZoneInfo) -> List[Match]:
    soup = BeautifulSoup(html, "lxml")
    tbl = _find_best_table(soup)
    if tbl is None:
        return []

    matches: List[Match] = []
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        cells = [td.get_text(" ", strip=True) for td in tds]
        joined = " ".join(cells).strip()
        if not joined:
            continue

        # locate score like "2 - 1" in any cell
        score_idx = None
        score_val = None
        for i, c in enumerate(cells):
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", c.strip())
            if m:
                score_idx = i
                score_val = (int(m.group(1)), int(m.group(2)))
                break
        if score_idx is None or score_val is None:
            continue

        # attempt datetime from early columns
        dt = None
        if len(cells) >= 2:
            dt = _parse_datetime(f"{cells[0]} {cells[1]}", tz)
        if dt is None:
            dt = _parse_datetime(joined, tz)
        if dt is None:
            continue

        # pick home/away by scanning around score position
        # common: ... home | score | away ...
        home = ""
        away = ""
        if score_idx >= 1:
            home = cells[score_idx - 1].strip()
        if score_idx + 1 < len(cells):
            away = cells[score_idx + 1].strip()

        # fallback: find two teamish strings
        if not home or not away or home == away:
            teamish = [c for c in cells if len(c) >= 3 and not re.search(r"\b\d{1,2}:\d{2}\b", c)]
            teamish = [c for c in teamish if not re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", c)]
            teamish = [c for c in teamish if not re.fullmatch(r"\d+\s*-\s*\d+", c)]
            if len(teamish) >= 2:
                home, away = teamish[0].strip(), teamish[1].strip()

        if not home or not away or home == away:
            continue

        matches.append(
            Match(
                kickoff_local=dt,
                home=home,
                away=away,
                competition="",
                result_home=score_val[0],
                result_away=score_val[1],
                status="Played",
                source="results",
            )
        )
    return matches


def match_key(m: Match) -> str:
    # Key used to merge results onto fixtures. We normalise date+teams.
    d = m.kickoff_local.date().isoformat()
    return f"{d}|{m.home.strip().lower()}|{m.away.strip().lower()}"


def merge(fixtures: List[Match], results: List[Match]) -> List[Match]:
    by_key: Dict[str, Match] = {match_key(m): m for m in fixtures}
    for r in results:
        k = match_key(r)
        if k in by_key:
            by_key[k].result_home = r.result_home
            by_key[k].result_away = r.result_away
            by_key[k].status = "Played"
        else:
            # result exists without fixture row (still add it)
            by_key[k] = r
    return sorted(by_key.values(), key=lambda x: x.kickoff_local)


# ----------------------------
# ICS + Index
# ----------------------------

def event_uid(team_name: str, m: Match) -> str:
    # stable UID per team+fixture (avoid duplicates)
    base = f"{team_name}|{m.kickoff_local.date().isoformat()}|{m.home}|{m.away}"
    return f"{sha1(base)}@football-fixtures-ics"


def build_team_calendar(team_name: str, matches: List[Match], div: DivisionConfig) -> str:
    tz = ZoneInfo(div.tz)
    cal = Calendar()
    cal.add("prodid", "-//football-fixtures-ics//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("method", "PUBLISH")
    cal.add("x-wr-calname", f"{team_name} Fixtures")
    cal.add("x-wr-timezone", div.tz)

    # rolling window (keeps files manageable, and avoids ancient seasons forever)
    start = datetime.now(tz).date() - timedelta(days=180)
    end = datetime.now(tz).date() + timedelta(days=365)

    for m in matches:
        if not (start <= m.kickoff_local.date() <= end):
            continue
        e = Event()
        e.add("uid", event_uid(team_name, m))
        e.add("dtstamp", now_utc())

        dtstart = m.kickoff_local
        dtend = dtstart + timedelta(minutes=div.default_duration_min)

        e.add("dtstart", dtstart)
        e.add("dtend", dtend)

        opponent = m.away if m.home.strip().lower() == team_name.strip().lower() else m.home
        ha = "vs" if m.home.strip().lower() == team_name.strip().lower() else "at"
        summary = f"{team_name} {ha} {opponent}".strip()

        # Result line
        result_line = ""
        if m.result_home is not None and m.result_away is not None:
            result_line = f"Result: {m.home} {m.result_home}–{m.result_away} {m.away}"

        desc_lines = []
        if result_line:
            desc_lines.append(result_line)
        if div.links.get("table"):
            desc_lines.append(f"Table: {div.links['table']}")
        if div.links.get("fixtures"):
            desc_lines.append(f"Fixtures: {div.links['fixtures']}")
        if div.links.get("results"):
            desc_lines.append(f"Results: {div.links['results']}")
        desc = "\n".join(desc_lines).strip()

        e.add("summary", summary)
        if desc:
            e.add("description", desc)
        if div.links.get("fixtures"):
            e.add("url", div.links["fixtures"])

        cal.add_component(e)

    return cal.to_ical().decode("utf-8")


def build_index(teams: List[str], repo_pages_base: str, div: DivisionConfig) -> str:
    teams_sorted = sorted(teams, key=lambda x: x.lower())
    items = []
    for t in teams_sorted:
        s = slugify(t)
        ics_url = f"{repo_pages_base}/{s}.ics"
        items.append(
            f"""
            <li>
              <strong>{t}</strong>
              — <a href="{ics_url}">Subscribe (ICS)</a>
            </li>
            """.strip()
        )

    table_url = div.links.get("table", "")
    fixtures_url = div.links.get("fixtures", "")
    results_url = div.links.get("results", "")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{div.name} — Calendars</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    li {{ margin: 10px 0; }}
    .links a {{ margin-right: 12px; }}
    code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{div.name} — Team Calendars</h1>
  <div class="links">
    {f'<a href="{table_url}">League Table</a>' if table_url else ''}
    {f'<a href="{fixtures_url}">Fixtures</a>' if fixtures_url else ''}
    {f'<a href="{results_url}">Results</a>' if results_url else ''}
  </div>
  <p>Subscribe URL format: <code>{repo_pages_base}/&lt;team-slug&gt;.ics</code></p>
  <ul>
    {"".join(items)}
  </ul>
</body>
</html>
""".strip()


# ----------------------------
# Diff + Telegram
# ----------------------------

def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def division_snapshot(matches: List[Match]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in matches:
        out.append({
            "kickoff_local": m.kickoff_local.isoformat(),
            "home": m.home,
            "away": m.away,
            "result_home": m.result_home,
            "result_away": m.result_away,
            "status": m.status,
        })
    return out


def diff_counts(prev: Any, curr: Any) -> Tuple[int, int]:
    if prev is None:
        return (0, len(curr) if curr else 0)
    prev_s = json.dumps(prev, sort_keys=True)
    curr_s = json.dumps(curr, sort_keys=True)
    if prev_s == curr_s:
        return (0, 0)
    return (1, 1)


def telegram_send(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception:
        pass


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ensure_dirs()
    divisions = load_divisions()

    # GitHub Pages base URL (best-effort for index link rendering)
    # If you later change username/repo, edit this once or compute it in CI.
    repo_pages_base = "https://justgeary.github.io/football-fixtures-ics"

    overall_status: Dict[str, Any] = {"generated_at_utc": now_utc().isoformat(), "divisions": []}

    for div in divisions:
        tz = ZoneInfo(div.tz)

        fixtures_url = div.links["fixtures"]
        results_url = div.links["results"]

        fx_resp = http_get(fixtures_url)
        rs_resp = http_get(results_url)

        fx_html = fx_resp.text if fx_resp is not None else ""
        rs_html = rs_resp.text if rs_resp is not None else ""

        if fx_resp.status_code != 200 or rs_resp.status_code != 200:
            Path(f"artifacts/{div.slug}.http_error.txt").write_text(
                f"Fixtures status: {fx_resp.status_code}\nResults status: {rs_resp.status_code}\n",
                encoding="utf-8",
            )
            telegram_send(f"⚠️ Calendar build failed ({div.name}): HTTP {fx_resp.status_code}/{rs_resp.status_code}")
            continue

        if looks_like_block_page(fx_html) or looks_like_block_page(rs_html):
            Path(f"artifacts/{div.slug}.blocked.html").write_text(fx_html[:5000], encoding="utf-8")
            telegram_send(f"⚠️ Calendar build blocked/denied ({div.name}). Not publishing.")
            continue

        fixtures = parse_fixtures(fx_html, tz)
        results = parse_results(rs_html, tz)
        merged = merge(fixtures, results)

        # Safe publish: require some matches
        if len(merged) == 0:
            Path(f"artifacts/{div.slug}.parse_failed.html").write_text(fx_html[:8000], encoding="utf-8")
            telegram_send(f"⚠️ Calendar build parsed 0 matches ({div.name}). Not publishing.")
            continue

        # Save snapshots + diff detection
        snap_path = Path(f"data/{div.slug}.latest.json")
        prev_path = Path(f"data/{div.slug}.prev.json")

        prev = load_json(snap_path)
        curr = division_snapshot(merged)

        # move latest → prev, then write latest
        if prev is not None:
            save_json(prev_path, prev)
        save_json(snap_path, curr)

        changed_flag, _ = diff_counts(prev, curr)

        # Determine team set
        teams = sorted({m.home for m in merged} | {m.away for m in merged})
        if div.include_teams:
            allowed = set(div.include_teams)
            teams = [t for t in teams if t in allowed]

        # Write ICS per team
        published = 0
        for team in teams:
            team_matches = [m for m in merged if m.home == team or m.away == team]
            ics = build_team_calendar(team, team_matches, div)
            if "BEGIN:VCALENDAR" not in ics:
                continue
            Path(f"docs/{slugify(team)}.ics").write_text(ics, encoding="utf-8")
            published += 1

        # Write index.html per division (single index for now)
        Path("docs/index.html").write_text(build_index(teams, repo_pages_base, div), encoding="utf-8")

        div_status = {
            "division": div.name,
            "slug": div.slug,
            "teams": len(teams),
            "matches": len(merged),
            "published_calendars": published,
            "last_success_utc": now_utc().isoformat(),
        }
        overall_status["divisions"].append(div_status)
        Path(f"docs/status_{div.slug}.json").write_text(json.dumps(div_status, indent=2), encoding="utf-8")

        if changed_flag:
            telegram_send(f"✅ Calendars updated: {div.name} — {published} team calendars")

    Path("docs/status.json").write_text(json.dumps(overall_status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
