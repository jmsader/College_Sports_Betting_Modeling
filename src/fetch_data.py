import sqlite3
import requests
from GLOBS import DB_SLUG, confirm_intent, trim_markets, INPUT_SLUG


def add_team_if_nseen(tid, name, teams_seen, market, conn):
    if tid not in teams_seen:
        teams_seen.add(tid)
        conn.execute(
            "INSERT OR IGNORE INTO teams (tid, name, mkt) VALUES (?, ?, ?);",
            (tid, name, market),
        )


confirm_intent()

API_KEY = None
with open(f"{INPUT_SLUG}api_key.txt", "r") as key_f:
    API_KEY = key_f.read().strip()

MARKETS = trim_markets()

with sqlite3.connect(DB_SLUG) as conn:
    conn.executescript(
        """
                        CREATE TABLE IF NOT EXISTS teams (
                                tid,
                                momentum_wins INTEGER DEFAULT 0,
                                momentum_losses INTEGER DEFAULT 0,
                                name TEXT,
                                mkt TEXT,
                                PRIMARY KEY (tid, mkt)
                        );
                            CREATE TABLE IF NOT EXISTS games (
                                gid INTEGER PRIMARY KEY,
                                unix INTEGER,
                                hid INTEGER NOT NULL,
                                aid INTEGER NOT NULL,
                                hodds INTEGER,
                                aodds INTEGER,
                                spread INTEGER,
                                result_tid INTEGER,
                                mkt TEXT,
                                FOREIGN KEY (hid) REFERENCES teams (tid),
                                FOREIGN KEY (aid) REFERENCES teams (tid),
                                FOREIGN KEY (result_tid) REFERENCES teams (tid)
                            );
                        """
    )

    NFL_RESP_FORMATTED_MKTS = ["nfl", "mlb", "nba", "nhl"]
    for market in MARKETS:
        is_nfl_resp_format = market in NFL_RESP_FORMATTED_MKTS
        year = 2025
        teams_seen = set()
        while True:
            if market != "nfl":
                req_str = f"https://api.sportsdata.io/v3/{market}/scores/json/Games/{year}?key={API_KEY}"
            else:
                req_str = f"https://api.sportsdata.io/v3/{market}/stats/json/ScoresFinal/{year}?key={API_KEY}"

            resp = requests.get(req_str).json()
            if type(resp) == dict:
                break
            for game in resp:
                if game.get("Status", "Final") != "Final":
                    continue
                add_team_if_nseen(
                    game["HomeTeamID"], game["HomeTeam"], teams_seen, market, conn
                )
                add_team_if_nseen(
                    game["AwayTeamID"], game["AwayTeam"], teams_seen, market, conn
                )
                result_team_id = None
                if market == "mlb":
                    score_suff = "TeamRuns"
                elif market == "nfl":
                    score_suff = "Score"
                else:
                    score_suff = "TeamScore"
                home_score_str = "Home" + score_suff
                away_score_str = "Away" + score_suff
                try:
                    if game[home_score_str] > game[away_score_str]:
                        result_team_id = game["HomeTeamID"]
                    elif game[home_score_str] < game[away_score_str]:
                        result_team_id = game["AwayTeamID"]
                    else:
                        continue
                except:
                    pass
                conn.execute(
                    "INSERT OR IGNORE INTO games (gid, unix, hid, aid, aodds, hodds, spread, result_tid, mkt) VALUES (?,?,?,?,?,?,?,?,?);",
                    (
                        game["GlobalGameID"],
                        game["DateTimeUTC"],
                        game["HomeTeamID"],
                        game["AwayTeamID"],
                        (
                            int(game["AwayTeamMoneyLine"])
                            if game.get("AwayTeamMoneyLine") is not None
                            else None
                        ),
                        (
                            int(game["HomeTeamMoneyLine"])
                            if game.get("HomeTeamMoneyLine") is not None
                            else None
                        ),
                        (
                            int(game.get("PointSpread"))
                            if game.get("PointSpread") is not None
                            else None
                        ),
                        result_team_id,
                        market,
                    ),
                )
            year -= 1
