DB_SLUG = "sql.db"
INPUT_SLUG = "../inputs/"


def trim_markets():
    with open(f"{INPUT_SLUG}ignored_markets", "r") as f:
        ignored_markets = [line.strip() for line in f.readlines()]

    return [mkt for mkt in MARKETS if mkt not in ignored_markets]


def confirm_intent():
    if input("ARE YOU SURE? (YES/NO)\n") != "YES":
        exit()


MARKETS = [
    "cbb",
    "cfb",
    "nfl",
    "mlb",
    "nba",
    "nhl",
]  # nfl not symetrical with rest of markets
