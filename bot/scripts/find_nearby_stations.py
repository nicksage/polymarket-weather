"""
find_nearby_stations.py — For each US Polymarket settlement station, find
all other Mesonet-tracked weather stations within a given radius.

The goal is a candidate pool for lead-lag analysis: stations upwind of the
settlement station may show temperature changes before the settlement
station does, providing an early signal for trading.

Data source: Iowa State Mesonet per-state network geojsons.  Queries:
  * State ASOS network for each US Polymarket station's state
  * State AWOS network (smaller airports / military)
  * Optionally other research networks (state mesonets) if --extended

For each candidate station, computes:
  * Great-circle distance (miles)
  * Bearing FROM the settlement station TO the candidate (degrees: 0=N, 90=E, ...)
  * Cardinal direction (N/NE/E/SE/S/SW/W/NW) — handy for "upwind when wind
    is from the X" reasoning later

Usage:
    cd bot
    python -m scripts.find_nearby_stations                       # default: 25 miles
    python -m scripts.find_nearby_stations --radius 50
    python -m scripts.find_nearby_stations --city Dallas Chicago
    python -m scripts.find_nearby_stations --csv data/nearby.csv
    python -m scripts.find_nearby_stations --extended            # include AWOS + research
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from collections import defaultdict

import httpx

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS  # type: ignore

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("nearby")
logging.getLogger("httpx").setLevel(logging.WARNING)

# US Polymarket cities → state network for ASOS lookups.
# (US_ASOS is the master network but its geojson is huge; per-state is faster.)
US_CITY_STATES: dict[str, str] = {
    "Atlanta":       "GA",
    "Austin":        "TX",
    "Chicago":       "IL",
    "Dallas":        "TX",
    "Denver":        "CO",
    "Houston":       "TX",
    "Los Angeles":   "CA",
    "Miami":         "FL",
    "NYC":           "NY",
    "San Francisco": "CA",
    "Seattle":       "WA",
}


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8   # earth radius, miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Bearing FROM point 1 TO point 2.  0=N, 90=E, 180=S, 270=W."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def cardinal(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((deg + 22.5) / 45) % 8]


def fetch_network(network: str, ttl_cache: dict | None = None) -> list[dict]:
    """Return list of station dicts {sid, name, lat, lon, elev, network}."""
    if ttl_cache is not None and network in ttl_cache:
        return ttl_cache[network]
    url = f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
    try:
        r = httpx.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        out = []
        for f in data.get("features", []):
            props = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [])
            if not coords or len(coords) < 2:
                continue
            out.append({
                "sid":     props.get("sid"),
                "name":    props.get("sname") or props.get("name"),
                "lat":     coords[1],
                "lon":     coords[0],
                "elev":    props.get("elevation"),
                "network": network,
            })
        if ttl_cache is not None:
            ttl_cache[network] = out
        return out
    except Exception as e:
        log.warning(f"Failed to fetch {network}: {e}")
        return []


def find_nearby(city: str, radius_mi: float, extended: bool,
                cache: dict) -> list[dict]:
    s = CITY_STATIONS.get(city)
    if not s:
        return []
    pm_icao, _, _, pm_lat, pm_lon = s
    state = US_CITY_STATES.get(city)
    if not state:
        return []   # non-US city

    # Networks to query for this city's state.
    networks = [f"{state}_ASOS"]
    if extended:
        networks.extend([f"{state}_AWOS", f"{state}_RWIS"])

    seen: set[str] = set()
    candidates: list[dict] = []
    for net in networks:
        for st in fetch_network(net, cache):
            if not st["sid"] or not st["lat"] or not st["lon"]:
                continue
            if st["sid"] in seen:
                continue
            seen.add(st["sid"])
            dist = haversine_miles(pm_lat, pm_lon, st["lat"], st["lon"])
            if dist > radius_mi:
                continue
            bear = bearing_deg(pm_lat, pm_lon, st["lat"], st["lon"])
            candidates.append({
                "polymarket_city":     city,
                "polymarket_station":  pm_icao,
                "sid":                 st["sid"],
                "name":                st["name"],
                "network":             st["network"],
                "lat":                 round(st["lat"], 4),
                "lon":                 round(st["lon"], 4),
                "elev_m":              st["elev"],
                "distance_mi":         round(dist, 2),
                "bearing_deg":         round(bear, 0),
                "direction":           cardinal(bear),
            })
    candidates.sort(key=lambda r: r["distance_mi"])
    return candidates


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--radius", type=float, default=25,
                   help="Search radius in miles (default: 25)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific US cities (default: all US)")
    p.add_argument("--extended", action="store_true",
                   help="Include AWOS + RWIS in addition to ASOS")
    p.add_argument("--csv", help="Write all rows to CSV at this path")
    args = p.parse_args()

    cities_filter = {c.lower() for c in args.city} if args.city else None
    cities = [c for c in US_CITY_STATES
              if cities_filter is None or c.lower() in cities_filter]

    if not cities:
        print(f"No US cities matched filter.  Available: {sorted(US_CITY_STATES)}")
        return 1

    log.info(f"Searching {len(cities)} US cities within {args.radius} mi"
             f"{' (extended networks)' if args.extended else ' (ASOS only)'}")

    cache: dict = {}
    all_rows: list[dict] = []
    for city in cities:
        s = CITY_STATIONS[city]
        rows = find_nearby(city, args.radius, args.extended, cache)
        all_rows.extend(rows)

        # Exclude the Polymarket station itself from the displayed neighbors
        neighbors = [r for r in rows if r["sid"] != s[0]
                                     and r["sid"] != s[0].lstrip("K")]

        print()
        print("=" * 86)
        print(f"  {city.upper()}  —  Polymarket station: {s[0]} ({s[3]:.3f}, {s[4]:.3f})")
        print(f"  {len(neighbors)} neighbor stations within {args.radius} mi")
        print("=" * 86)
        if not neighbors:
            print("  (none found — try larger --radius or --extended)")
            continue
        print(f"  {'distance':>9} {'dir':>3} {'bearing':>7}  "
              f"{'sid':<6} {'network':<10} {'elev':>6}  name")
        for r in neighbors:
            elev = f"{r['elev_m']:.0f}m" if r['elev_m'] is not None else "  ?"
            print(f"  {r['distance_mi']:>6.1f} mi  {r['direction']:>3} "
                  f"{r['bearing_deg']:>5.0f}°   {r['sid']:<6} "
                  f"{r['network']:<10} {elev:>6}  "
                  f"{(r['name'] or '')[:42]}")

    print()
    print("=" * 86)
    print(f"  TOTAL: {len(all_rows)} (city, neighbor) pairs across "
          f"{len(cities)} cities")
    print("=" * 86)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        fields = ["polymarket_city", "polymarket_station", "sid", "name",
                  "network", "lat", "lon", "elev_m",
                  "distance_mi", "bearing_deg", "direction"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"\nWrote {len(all_rows)} rows to {args.csv}")

    print()
    print("Reading the bearing column:")
    print("  0° = N, 90° = E, 180° = S, 270° = W")
    print("  Direction is FROM the Polymarket station TO the neighbor.")
    print("  So 'NW' direction = neighbor is northwest of the airport.")
    print("  When the prevailing wind is FROM the NW (i.e. blowing toward SE),")
    print("  that NW neighbor is UPWIND of the Polymarket station — and a")
    print("  candidate early indicator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())