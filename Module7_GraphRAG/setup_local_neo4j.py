"""Prepare the local Neo4j instance for the lesson notebooks.

- Default database (NEO4J_DATABASE, used by L2-L6): loaded with the classic movies graph.
- SEC database (NEO4J_SEC_DATABASE, used by L7): loaded from data/neo4j-sec-sample.dump, then
  enriched with Address nodes (offline geocoding via GeoNames) and full-text indexes,
  which is what the course's pre-built Lesson 7 database contained.

Usage:
    uv run python setup_local_neo4j.py           # load whatever is missing
    uv run python setup_local_neo4j.py --reset   # wipe the default database first, then reload movies
"""
import csv
import html
import io
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import truststore
from dotenv import load_dotenv
from neo4j import GraphDatabase

truststore.inject_into_ssl()
load_dotenv(".env", override=True)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GEO = DATA / "geo"
MOVIES_SCRIPT = DATA / "movies.cypher"
SEC_DUMP = DATA / "neo4j-sec-sample.dump"

DB = os.getenv("NEO4J_DATABASE") or "neo4j"
SEC_DB = os.getenv("NEO4J_SEC_DATABASE") or "sec"
DESKTOP = Path.home() / ".Neo4jDesktop2"

# Headquarters addresses from each company's 2023 Form 10-K, in the same
# "street, CITY, STATE-CODE, POSTAL" layout that SEC uses for manager addresses.
COMPANY_HQ = {
    "64110D": "3060 Olsen Drive, San Jose, CA, 95128",                  # NetApp
    "697435": "3000 Tannery Way, Santa Clara, CA, 95054",               # Palo Alto Networks
    "31428X": "942 South Shady Grove Road, Memphis, TN, 38120",         # FedEx
    "36241U": "1213 Elko Drive, Sunnyvale, CA, 94089",                  # GSI Technology
    "65249B": "1211 Avenue of the Americas, New York, NY, 10036",       # News Corp
    "654106": "One Bowerman Drive, Beaverton, OR, 97005",               # Nike
    "958102": "5601 Great Oaks Parkway, San Jose, CA, 95119",           # Western Digital
    "G7945J": "38/39 Fitzwilliam Square, Dublin, L2, D02 NX53",         # Seagate
    "G06242": "350 Bush Street, Floor 13, San Francisco, CA, 94104",    # Atlassian
    "003783": "One Apple Park Way, Cupertino, CA, 95014",               # Apple
}

COUNTRY_NAME_FIXES = {
    "KOREA, REPUBLIC OF": "South Korea",
    "CZECH REPUBLIC": "Czechia",
    "HONG KONG": "Hong Kong",
    "UNITED KINGDOM": "United Kingdom",
}

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    notifications_min_severity="OFF",
)
driver.verify_connectivity()


def run(query, db=DB, **params):
    records, _, _ = driver.execute_query(query, params, database_=db)
    return records


def download(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        # sec.gov rejects requests whose User-Agent lacks a contact address
        req = urllib.request.Request(url, headers={"User-Agent": "knowledge-graph-course setup admin@example.com"})
        dest.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    return dest


# ---------------------------------------------------------------- default database

def reset_default_db():
    print(f"Resetting database '{DB}'...")
    # CALL { } IN TRANSACTIONS needs an auto-commit transaction, which session.run provides
    with driver.session(database=DB) as session:
        session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS").consume()
    for r in run("SHOW CONSTRAINTS YIELD name"):
        run(f"DROP CONSTRAINT `{r['name']}` IF EXISTS")
    for r in run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'"):
        run(f"DROP INDEX `{r['name']}` IF EXISTS")


def load_movies():
    if run("MATCH (m:Movie) RETURN count(m) AS c")[0]["c"] > 0:
        print(f"Movies graph already present in '{DB}'.")
        return
    statements = [s.strip() for s in MOVIES_SCRIPT.read_text(encoding="utf-8").split(";") if s.strip()]
    # The course's movie database had no constraints; L2 re-creates Person nodes with CREATE.
    statements = [s for s in statements if not s.upper().startswith("CREATE CONSTRAINT")]
    for s in statements:
        run(s)
    print(f"Loaded movies graph into '{DB}': {run('MATCH (n) RETURN count(n) AS c')[0]['c']} nodes.")


# ---------------------------------------------------------------- SEC database

def find_dbms_home():
    if os.getenv("NEO4J_HOME"):
        return Path(os.environ["NEO4J_HOME"])
    homes = [p for p in (DESKTOP / "Data" / "dbmss").glob("dbms-*") if (p / "bin").exists()]
    if len(homes) != 1:
        sys.exit("Set NEO4J_HOME to the Neo4j instance folder (found %d Desktop instances)." % len(homes))
    return homes[0]


def find_java21():
    if os.getenv("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    runtimes = sorted((DESKTOP / "Cache" / "runtime").glob("zulu21*"))
    if not runtimes:
        sys.exit("Set JAVA_HOME to a Java 21 runtime (needed by neo4j-admin).")
    return str(runtimes[-1])


def load_sec_database():
    if SEC_DB in {r["name"] for r in run("SHOW DATABASES YIELD name", db="system")}:
        print(f"Database '{SEC_DB}' already exists.")
    else:
        home = find_dbms_home()
        staging = DATA / ".dump-staging"
        staging.mkdir(exist_ok=True)
        (staging / f"{SEC_DB}.dump").write_bytes(SEC_DUMP.read_bytes())
        admin = home / "bin" / ("neo4j-admin.bat" if os.name == "nt" else "neo4j-admin")
        print(f"Loading {SEC_DUMP.name} into '{SEC_DB}' with neo4j-admin...")
        subprocess.run(
            [str(admin), "database", "load", SEC_DB, f"--from-path={staging}", "--overwrite-destination=true"],
            check=True, env={**os.environ, "JAVA_HOME": find_java21()}, stdout=subprocess.DEVNULL,
        )
        (staging / f"{SEC_DB}.dump").unlink()
        staging.rmdir()
        run(f"CREATE DATABASE `{SEC_DB}` IF NOT EXISTS WAIT", db="system")
    for _ in range(60):
        status = run("SHOW DATABASE $name YIELD currentStatus, statusMessage", db="system", name=SEC_DB)[0]
        if status["currentStatus"] == "online":
            break
        time.sleep(2)
    else:
        sys.exit(f"Database '{SEC_DB}' is not online: {status['statusMessage']}")


def load_geonames():
    """Lookup tables: US ZIP -> place, Canadian FSA -> place, (city, country) -> place."""
    def read_zip(url, name):
        with zipfile.ZipFile(download(url, GEO / Path(url).name)) as z:
            return list(csv.reader(io.TextIOWrapper(z.open(name), encoding="utf-8"), delimiter="\t"))

    # zip -> (place, state name, state code, lat, lon)
    us_zip = {r[1]: (r[2], r[3], r[4], float(r[9]), float(r[10]))
              for r in read_zip("https://download.geonames.org/export/zip/US.zip", "US.txt")}
    # forward sortation area (first 3 chars of the postal code) -> (province, lat, lon)
    ca_fsa = {r[1][:3].upper(): (r[3], float(r[9]), float(r[10]))
              for r in read_zip("https://download.geonames.org/export/zip/CA.zip", "CA.txt")}

    countries = {}
    info = download("https://download.geonames.org/export/dump/countryInfo.txt", GEO / "countryInfo.txt")
    for line in info.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            f = line.split("\t")
            countries[f[4].upper()] = (f[0], f[4])

    # (NAME, country) and (NAME, country, admin1) -> most populous matching city (name, (lat, lon), pop)
    cities = {}
    for r in read_zip("https://download.geonames.org/export/dump/cities15000.zip", "cities15000.txt"):
        name, ascii_name, alt, cc, admin1, pop = r[1], r[2], r[3], r[8], r[10], int(r[14] or 0)
        for key in {name.upper(), ascii_name.upper(), *(a.upper() for a in alt.split(",") if a)}:
            for k in ((key, cc), (key, cc, admin1)):
                best = cities.get(k)
                if best is None or pop > best[2]:
                    cities[k] = (name, (float(r[4]), float(r[5])), pop)
    return us_zip, ca_fsa, countries, cities


def load_sec_region_codes():
    page = download("https://www.sec.gov/submit-filings/filer-support-resources/edgar-state-country-codes",
                    GEO / "edgar-state-country-codes.html").read_text(encoding="utf-8")
    rows = re.findall(r"<tr[^>]*>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>", page, re.S)
    clean = lambda s: html.unescape(re.sub("<.*?>", "", s)).strip()
    return {clean(a): clean(b) for a, b in rows}


def geocode(address, us_zip, ca_fsa, countries, cities, codes):
    """Return (city, state, country, lat, lon, is_city_centre) for an SEC-style address, or None."""
    parts = [p.strip() for p in (address or "").split(",") if p.strip()]
    if len(parts) < 3:
        return None
    city, code, postal = parts[-3], parts[-2].upper(), parts[-1].upper()
    region = codes.get(code)
    if region is None:
        return None
    if code.isalpha():
        hit = us_zip.get(re.sub(r"\D", "", postal)[:5])
        if not hit:
            return None
        place, state, state_code, lat, lon = hit
        centre = cities.get((place.upper(), "US", state_code))
        if centre:
            return (place, state, "United States", *centre[1], True)
        return (place, state, "United States", lat, lon, False)
    if region.endswith(", CANADA"):
        hit = ca_fsa.get(postal.replace(" ", "")[:3])
        if not hit:
            return None
        province, lat, lon = hit
        centre = cities.get((city.upper(), "CA"))
        if centre:
            return (centre[0], province, "Canada", *centre[1], True)
        return (city.title(), province, "Canada", lat, lon, False)
    country_name = COUNTRY_NAME_FIXES.get(region, region.title())
    iso = countries.get(country_name.upper(), (None,))[0]
    for candidate in [city, *reversed(parts[:-3])]:
        hit = cities.get((candidate.upper(), iso))
        if hit:
            return (hit[0], country_name, country_name, *hit[1], True)
    return None


def enrich_sec_database():
    if run("MATCH (a:Address) RETURN count(a) AS c", db=SEC_DB)[0]["c"] > 0:
        print(f"'{SEC_DB}' already has Address nodes.")
        return
    print("Geocoding manager and company addresses (offline, GeoNames)...")
    us_zip, ca_fsa, countries, cities = load_geonames()
    codes = load_sec_region_codes()

    for cusip6, address in COMPANY_HQ.items():
        run("MATCH (c:Company {cusip6: $cusip6}) SET c.companyAddress = $address",
            db=SEC_DB, cusip6=cusip6, address=address)

    rows = []
    for r in run("MATCH (m:Manager) RETURN 'Manager' AS label, m.managerCik AS key, m.managerAddress AS address "
                 "UNION ALL MATCH (c:Company) RETURN 'Company' AS label, c.cusip6 AS key, c.companyAddress AS address",
                 db=SEC_DB):
        hit = geocode(r["address"], us_zip, ca_fsa, countries, cities, codes)
        if hit:
            city, state, country, lat, lon, is_centre = hit
            rows.append({"label": r["label"], "key": r["key"], "city": city, "state": state,
                         "country": country, "lat": lat, "lon": lon, "is_centre": is_centre})

    # One Address node per (city, state), located at the city centre when GeoNames knows the
    # city, otherwise at the mean of its members' postal-code centroids.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["city"], row["state"])].append(row)
    for members in groups.values():
        centres = [m for m in members if m["is_centre"]] or members
        lat = statistics.fmean(m["lat"] for m in centres)
        lon = statistics.fmean(m["lon"] for m in centres)
        for m in members:
            m["lat"], m["lon"] = lat, lon
            del m["is_centre"]

    run("CREATE INDEX composite_address_index IF NOT EXISTS FOR (a:Address) ON (a.city, a.state)", db=SEC_DB)
    run("CREATE POINT INDEX address_locations IF NOT EXISTS FOR (a:Address) ON (a.location)", db=SEC_DB)
    run("""
        UNWIND $rows AS row
        MERGE (addr:Address {city: row.city, state: row.state})
          ON CREATE SET addr.country = row.country,
                        addr.location = point({latitude: row.lat, longitude: row.lon})
        WITH addr, row
        CALL (addr, row) {
          WITH addr, row WHERE row.label = 'Manager'
          MATCH (m:Manager {managerCik: row.key})
          MERGE (m)-[:LOCATED_AT]->(addr)
          UNION
          WITH addr, row WHERE row.label = 'Company'
          MATCH (c:Company {cusip6: row.key})
          MERGE (c)-[:LOCATED_AT]->(addr)
        }
        """, db=SEC_DB, rows=rows)

    run("CREATE FULLTEXT INDEX fullTextManagerNames IF NOT EXISTS FOR (m:Manager) ON EACH [m.managerName]",
        db=SEC_DB)
    run("CREATE FULLTEXT INDEX fullTextCompanyNames IF NOT EXISTS FOR (c:Company) ON EACH [c.companyName]",
        db=SEC_DB)
    run("CALL db.awaitIndexes(300)", db=SEC_DB)

    total = run("MATCH (n) WHERE n:Manager OR n:Company RETURN count(n) AS c", db=SEC_DB)[0]["c"]
    print(f"Located {len(rows)} of {total} managers/companies at "
          f"{run('MATCH (a:Address) RETURN count(a) AS c', db=SEC_DB)[0]['c']} addresses.")


if "--reset" in sys.argv:
    reset_default_db()
load_movies()
load_sec_database()
enrich_sec_database()
labels = run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC", db=SEC_DB)
print(f"'{SEC_DB}' contents: " + ", ".join(f"{r['label']}={r['c']}" for r in labels))
driver.close()
