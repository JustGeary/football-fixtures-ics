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
from dateutil import parser as dtparser
from icalendar import Calendar, Event


# ----------------------------
# Models
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
    status: str = "Scheduled"   # Scheduled / Played / Postponed / Home Walkover / Away Walkover
    source: str = ""            # fixtures/results
    match_type: str = ""        # "L", "CC"
    raw_status: str = ""        # "Postponed", "Home Walkover", etc
    score_text: str = ""        # e.g. "2 - 7 (HT 1-2)" or "Postponed P - P"
    fixture_id: str = ""        # e.g. "fixture-29407768"


# ----------------------------
# Basic helpers
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


def ensure_dirs() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)


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
                slug=raw["slug"],
                name=raw["name"],
                tz=raw.get("timezone", "Europe/London"),
                default_duration_min=int(raw.get("default_duration_min", 120)),
                links=links,
                include_teams=raw.get("include_teams"),
            )
        )

    if not divs:
        raise RuntimeError("No divisions/*.yml found")

    return divs


def http_get(url: str, *, timeout: int = 45) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; football-fixtures-ics/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(7):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** attempt) + 0.25)
                continue
            return resp
        except Exception as e:
            last_exc = e
            time.sleep((2 ** attempt) + 0.35)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def looks_like_block_page(html: str) -> bool:
    low = (html or "").lower()
    return ("cloudflare" in low and "attention required" in low) or ("access denied" in low) or ("forbidden" in low)


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
# Normalisation / classification
# ----------------------------

def norm_ws(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def is_noise_cell(text: str) -> bool:
    t = norm_ws(text)
    if not t:
        return True
    u = t.upper()

    if u in {"L", "CC"}:
        return True

    if re.match(r"^\(.*\)$", t):  # (HT 1-2) etc
        return True

    if re.fullmatch(r"[PHAW]\s*-\s*[PHAW]", u):  # P - P, H - W, A - W
        return True

    return False


def looks_like_competition(text: str) -> bool:
    t = norm_ws(text).lower()
    if not t:
        return False
    if any(w in t for w in ["division", "cup", "league"]):
        return True
    if re.match(r"^u\d{1,2}\b", t):
        return True
    return False


def is_teamish(text: str) -> bool:
    t = norm_ws(text)
    if not t or is_noise_cell(t):
        return False

    if _DATE_TIME_RE.search(t):
        return False

    if re.fullmatch(r"\d+\s*-\s*\d+", t):
        return False

    if detect_status_cell(t):
        return False

    # key safety: don't treat competitions as teams
    if looks_like_competition(t):
        return False

    if not re.search(r"[A-Za-z]", t):
        return False

    return len(t) >= 3


def extract_row_datetime(cells: List[str], tz: ZoneInfo) -> Optional[datetime]:
    for c in cells:
        c = norm_ws(c)
        m = _DATE_TIME_RE.search(c)
        if m:
            try:
                dt = dtparser.parse(m.group(0), dayfirst=True, fuzzy=False)
                return dt.replace(tzinfo=tz)
            except Exception:
                pass

    try:
        dt = dtparser.parse(" ".join(norm_ws(x) for x in cells), dayfirst=True, fuzzy=True)
        return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)
    except Exception:
        return None


# ----------------------------
# Fixtures table selection
# ----------------------------

def _find_best_table(soup: BeautifulSoup):
    tables = soup.find_all("table")
    if not tables:
        return None

    def score_table(tbl) -> int:
        th_text = " ".join(th.get_text(" ", strip=True).lower() for th in tbl.find_all("th"))
        score = 0
        if "home" in th_text:
            score += 4
        if "away" in th_text:
            score += 4
        if "date" in th_text:
            score += 3
        if "time" in th_text:
            score += 3

        body_text = norm_ws(tbl.get_text(" ", strip=True))
        score += 10 * len(_DATE_TIME_RE.findall(body_text))
        score += min(len(tbl.find_all("tr")), 500) // 5
        return score

    return max(tables, key=score_table)


# ----------------------------
# Parsers
# ----------------------------

def parse_fixtures(html: str, tz: ZoneInfo, division_name: str) -> List[Match]:
    soup = BeautifulSoup(html, "lxml")
    tbl = _find_best_table(soup)
    if tbl is None:
        return []

    out: List[Match] = []

    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue

        cells = [norm_ws(td.get_text(" ", strip=True)) for td in tds]
        if not " ".join(cells).strip():
            continue

        # ✅ NEW: capture match type from the first column (L / CC)
        match_type = ""
        if cells:
            first = cells[0].strip().upper()
            if first in {"L", "CC"}:
                match_type = first

        dt = extract_row_datetime(cells, tz)
        if dt is None:
            continue

        teams = [normalise_team_name(c) for c in cells if is_teamish(c)]
        teams = [t for t in teams if t]
        if len(teams) < 2:
            continue

        home = teams[0]
        away = ""
        for t in teams[1:]:
            if t.lower() != home.lower():
                away = t
                break
        if not away:
            continue

        raw_status = ""
        status = "Scheduled"
        for c in cells:
            s = detect_status_cell(c)
            if s:
                raw_status = s
                status = s
                break

        out.append(
            Match(
                kickoff_local=dt,
                home=home,
                away=away,
                competition=division_name,
                status=status,
                raw_status=raw_status,
                match_type=match_type,  # ✅ NEW
                source="fixtures",
            )
        )

    return out


_SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def _extract_score(score_text: str) -> Tuple[Optional[int], Optional[int]]:
    m = _SCORE_RE.match(norm_ws(score_text))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_results_divs(html: str, tz: ZoneInfo, division_name: str) -> List[Match]:
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.select_one("div.results-table-2 div.tbody")
    if not tbody:
        return []

    blocks = tbody.select("div[id^=fixture-]")
    out: List[Match] = []

    for fx in blocks:
        fixture_id = fx.get("id", "") or ""

        # Type: "L" or "CC ..." (often includes cup name in the same text)
        type_el = fx.select_one(".type-col a, .type-col p, .type-col")
        type_text = norm_ws(type_el.get_text(" ", strip=True)) if type_el else ""
        match_type = ""
        if type_text.upper().startswith("CC"):
            match_type = "CC"
        elif type_text.upper().startswith("L"):
            match_type = "L"

        # Datetime
        dt_el = fx.select_one(".datetime-col a, .datetime-col")
        dt_txt = norm_ws(dt_el.get_text(" ", strip=True)) if dt_el else ""
        if not dt_txt:
            continue

        try:
            dt = dtparser.parse(dt_txt, dayfirst=True, fuzzy=True).replace(tzinfo=tz)
        except Exception:
            continue

        # Teams
        home_el = fx.select_one(".home-team-col .team-name a, .home-team-col a, .home-team-col")
        away_el = fx.select_one(".road-team-col .team-name a, .road-team-col a, .road-team-col")
        home = normalise_team_name(home_el.get_text(" ", strip=True)) if home_el else ""
        away = normalise_team_name(away_el.get_text(" ", strip=True)) if away_el else ""
        if not home or not away:
            continue

        # Competition
        comp_el = fx.select_one(".fg-col p, .fg-col")
        comp = norm_ws(comp_el.get_text(" ", strip=True)) if comp_el else ""
        if not comp:
            comp = division_name

        # Score / Status
        score_el = fx.select_one(".score-col")
        score_text_full = norm_ws(score_el.get_text(" ", strip=True)) if score_el else ""

        status = "Scheduled"
        raw_status = ""
        result_home = None
        result_away = None

        if score_text_full:
            low = score_text_full.lower()
            if low.startswith("postponed"):
                status = "Postponed"
                raw_status = "Postponed"
            elif low.startswith("home walkover"):
                status = "Home Walkover"
                raw_status = "Home Walkover"
            elif low.startswith("away walkover"):
                status = "Away Walkover"
                raw_status = "Away Walkover"
            else:
                # Often: "0 - 2 (HT 0-0)" or "3 - 0"
                first_part = score_text_full.split("(", 1)[0].strip()
                rh, ra = _extract_score(first_part)
                if rh is not None and ra is not None:
                    result_home, result_away = rh, ra
                    status = "Played"

        out.append(
            Match(
                kickoff_local=dt,
                home=home,
                away=away,
                competition=comp,
                result_home=result_home,
                result_away=result_away,
                status=status,
                raw_status=raw_status,
                score_text=score_text_full,
                match_type=match_type,
                source="results",
                fixture_id=fixture_id,
            )
        )

    return out


# ----------------------------
# Merge
# ----------------------------

def match_key(m: Match) -> str:
    dt = m.kickoff_local.replace(second=0, microsecond=0).isoformat()
    home = normalise_team_name(m.home).lower()
    away = normalise_team_name(m.away).lower()
    comp = (m.competition or "").strip().lower()
    return f"{dt}|{home}|{away}|{comp}"


def merge(fixtures: List[Match], results: List[Match], division_name: str) -> List[Match]:
    # ensure fixtures comp stays stable
    for f in fixtures:
        f.competition = division_name

    by_id: Dict[str, Match] = {}
    by_key: Dict[str, Match] = {}

    for f in fixtures:
        if f.fixture_id:
            by_id[f.fixture_id] = f
        by_key[match_key(f)] = f

    for r in results:
        merged = False

        if r.fixture_id and r.fixture_id in by_id:
            tgt = by_id[r.fixture_id]
            tgt.result_home = r.result_home
            tgt.result_away = r.result_away
            tgt.status = r.status or tgt.status
            tgt.raw_status = r.raw_status
            tgt.match_type = r.match_type
            tgt.score_text = r.score_text
            merged = True

        if not merged:
            k = match_key(r)
            if k in by_key:
                tgt = by_key[k]
                tgt.result_home = r.result_home
                tgt.result_away = r.result_away
                tgt.status = r.status or tgt.status
                tgt.raw_status = r.raw_status
                tgt.match_type = r.match_type
                tgt.score_text = r.score_text
            else:
                by_key[k] = r  # results-only row

    return sorted(by_key.values(), key=lambda x: x.kickoff_local)


# ----------------------------
# ICS
# ----------------------------

def event_uid(team_name: str, m: Match) -> str:
    base = (
        f"{team_name}|{m.kickoff_local.replace(second=0, microsecond=0).isoformat()}"
        f"|{m.home}|{m.away}|{m.competition}"
    )
    return f"{sha1(base)}@football-fixtures-ics"


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

        is_home = normalise_team_name(m.home).lower() == team_norm
        opponent = m.away if is_home else m.home
        ha = "vs" if is_home else "at"

        summary = f"{team_name} {ha} {opponent}".strip()
        if m.status in {"Postponed", "Home Walkover", "Away Walkover"}:
            summary = f"[{m.status}] {summary}"
        e.add("summary", summary)

        desc: List[str] = []
        if m.competition:
            desc.append(f"Competition: {m.competition}")
        if m.match_type:
            desc.append(f"Type: {m.match_type}")

        if m.result_home is not None and m.result_away is not None:
            desc.append(f"Result: {m.home} {m.result_home}–{m.result_away} {m.away}")
            if m.score_text and "(" in m.score_text:
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
# Index
# ----------------------------

def guess_pages_base() -> str:
    env_base = os.getenv("PAGES_BASE_URL", "").strip()
    if env_base:
        return env_base.rstrip("/")

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if repo and "/" in repo:
        owner, repo_name = repo.split("/", 1)
        return f"https://{owner}.github.io/{repo_name}"

    return "https://justgeary.github.io/football-fixtures-ics"


def build_index(teams: List[str], repo_pages_base: str, div: DivisionConfig) -> str:
    teams_sorted = sorted(teams, key=lambda x: x.lower())
    items = []
    for t in teams_sorted:
        s = slugify(t)
        ics_url = f"{repo_pages_base}/{s}.ics"
        items.append(f'<li><strong>{t}</strong> — <a href="{ics_url}">Subscribe (ICS)</a></li>')

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
</html>"""


# ----------------------------
# Snapshot helpers
# ----------------------------

def division_snapshot(matches: List[Match]) -> List[Dict[str, Any]]:
    return [{
        "kickoff_local": m.kickoff_local.isoformat(),
        "home": m.home,
        "away": m.away,
        "competition": m.competition,
        "result_home": m.result_home,
        "result_away": m.result_away,
        "status": m.status,
        "match_type": m.match_type,
        "raw_status": m.raw_status,
        "score_text": m.score_text,
        "fixture_id": m.fixture_id,
        "source": m.source,
    } for m in matches]


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def snapshots_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ensure_dirs()
    divisions = load_divisions()
    pages_base = guess_pages_base()

    for div in divisions:
        tz = ZoneInfo(div.tz)
        division_name = div.name

        fixtures_url = div.links["fixtures"]
        results_url = div.links["results"]

        fx_resp = http_get(fixtures_url)
        rs_resp = http_get(results_url)

        if fx_resp.status_code != 200 or rs_resp.status_code != 200:
            telegram_send(f"⚠️ HTTP error ({div.name}): fixtures {fx_resp.status_code}, results {rs_resp.status_code}")
            continue

        fx_html = fx_resp.text
        rs_html = rs_resp.text

        if looks_like_block_page(fx_html) or looks_like_block_page(rs_html):
            telegram_send(f"⚠️ Blocked/denied by Full-Time ({div.name}). Not publishing.")
            continue

        fixtures = parse_fixtures(fx_html, tz, division_name=division_name)
        results = parse_results_divs(rs_html, tz, division_name=division_name)

        # Debug artifact
        Path(f"artifacts/{div.slug}.results_count.txt").write_text(
            f"results_parsed={len(results)}\n",
            encoding="utf-8",
        )

        merged = merge(fixtures, results, division_name=division_name)

        if not merged:
            telegram_send(f"⚠️ Parsed 0 matches ({div.name}). Not publishing.")
            continue

        snap_path = Path(f"data/{div.slug}.latest.json")
        prev = load_json(snap_path)
        curr = division_snapshot(merged)
        save_json(snap_path, curr)

        changed = not snapshots_equal(prev, curr)

        # ✅ KEY FIX: Only publish calendars for teams in LEAGUE fixtures (type "L")
        league_only = [m for m in fixtures if (m.match_type or "").upper() == "L"]
        league_teams = sorted({m.home for m in league_only} | {m.away for m in league_only})
        teams = league_teams

        if div.include_teams:
            allowed = set(div.include_teams)
            teams = [t for t in teams if t in allowed]

        published = 0
        for team in teams:
            team_matches = [m for m in merged if m.home == team or m.away == team]
            ics_bytes = build_team_calendar_bytes(team, team_matches, div)
            Path(f"docs/{slugify(team)}.ics").write_bytes(ics_bytes)
            published += 1

        Path("docs/index.html").write_text(build_index(teams, pages_base, div), encoding="utf-8")

        if changed:
            telegram_send(f"✅ Calendars updated: {div.name} — {published} team calendars")


if __name__ == "__main__":
    main()
