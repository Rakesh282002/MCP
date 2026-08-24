"""
MCP Server — Web Search + Weather Forecast tools

Run: python step1_mcp_server.py
Dependencies: pip install mcp httpx beautifulsoup4
"""

import sys
import site
from pathlib import Path

# === Fix: repo folder "mcp" shadows the installed mcp package ===
# Strategy: temporarily nuke sys.path, import mcp from site-packages, then restore.
_this_dir = str(Path(__file__).resolve().parent)

# 1. Remove conflicting paths
_original_path = sys.path[:]
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != _this_dir
            and str(Path(p, "mcp").resolve()) != _this_dir
            and p != ""]

# 2. Ensure site-packages are present
for sp in site.getsitepackages():
    if sp not in sys.path:
        sys.path.insert(0, sp)
try:
    user_sp = site.getusersitepackages()
    if user_sp not in sys.path:
        sys.path.insert(0, user_sp)
except Exception:
    pass
# === End fix ===

import os
import httpx
from mcp.server.fastmcp import FastMCP

MCP_PORT = int(os.environ.get("MCP_PORT", 8080))
mcp = FastMCP("Search & Weather", host="0.0.0.0", port=MCP_PORT)


# ---------------------------------------------------------------------------
# TOOL 1: Web Search (DuckDuckGo — no API key needed)
# ---------------------------------------------------------------------------
@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return top results.

    Args:
        query: The search query (e.g., "Python 3.14 new features")
        max_results: Number of results to return (default 5, max 10)

    Returns:
        Search results with title, URL, and snippet.
    """
    max_results = min(max_results, 10)
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        response = httpx.post(url, data={"q": query}, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Error performing search: {str(e)}"

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for i, result in enumerate(soup.select(".result"), 1):
        if i > max_results:
            break
        title_elem = result.select_one(".result__title a")
        snippet_elem = result.select_one(".result__snippet")
        title = title_elem.get_text(strip=True) if title_elem else "No title"
        link = title_elem.get("href", "") if title_elem else ""
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else "No description"
        results.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")

    if not results:
        return f"No results found for: '{query}'"

    return f"Search results for '{query}':\n\n" + "\n\n".join(results)


# ---------------------------------------------------------------------------
# TOOL 2: Weather Forecast (Open-Meteo API — free, no API key needed)
# ---------------------------------------------------------------------------
@mcp.tool()
def weather_forecast(city: str) -> str:
    """
    Get the current weather and 3-day forecast for a city.

    Args:
        city: City name (e.g., "London", "New York", "Hyderabad")

    Returns:
        Current temperature, conditions, humidity, wind, and 3-day forecast.
    """
    # Geocode city name to lat/lon
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        geo_resp = httpx.get(geo_url, params={"name": city, "count": 1}, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
    except Exception as e:
        return f"Error geocoding city '{city}': {str(e)}"

    if not geo_data.get("results"):
        return f"City '{city}' not found. Try a different spelling."

    location = geo_data["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]
    full_name = f"{location['name']}, {location.get('admin1', '')}, {location.get('country', '')}"

    # Get weather from Open-Meteo
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 3,
    }

    try:
        weather_resp = httpx.get(weather_url, params=params, timeout=10)
        weather_resp.raise_for_status()
        data = weather_resp.json()
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

    current = data.get("current", {})
    temp = current.get("temperature_2m", "N/A")
    humidity = current.get("relative_humidity_2m", "N/A")
    wind = current.get("wind_speed_10m", "N/A")
    code = current.get("weather_code", -1)
    condition = _weather_code_to_text(code)

    output = (
        f"Weather for {full_name}:\n\n"
        f"Current Conditions:\n"
        f"  Temperature: {temp}°C\n"
        f"  Condition:   {condition}\n"
        f"  Humidity:    {humidity}%\n"
        f"  Wind Speed:  {wind} km/h\n"
    )

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    maxs = daily.get("temperature_2m_max", [])
    mins = daily.get("temperature_2m_min", [])
    codes = daily.get("weather_code", [])
    precip = daily.get("precipitation_sum", [])

    if dates:
        output += "\n3-Day Forecast:\n"
        for i, date in enumerate(dates):
            cond = _weather_code_to_text(codes[i]) if i < len(codes) else "Unknown"
            output += f"  {date}: {mins[i]}°C — {maxs[i]}°C, {cond}, Rain: {precip[i]}mm\n"

    return output


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to readable text."""
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    return codes.get(code, f"Unknown (code {code})")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"  MCP Server — http://0.0.0.0:{MCP_PORT}")
    print("  Tools: web_search, weather_forecast")
    print("=" * 60)
    mcp.run(transport="sse")
