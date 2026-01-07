import sqlite3
import pandas as pd
from GLOBS import DB_SLUG, trim_markets
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path


# tunable parameters
MOM_TOLS = [1, 2, 3]
BET_STREAK_EXPS = [0.0, 0.5, 1.0]
WIN_STRATS = [True, False]
BET_AGAINST_MOM = [True, False]

VERIFY_PROP = 1 / 4.0
TRAINING_RAND_ITS = int(
    input("Enter the number of training iterations (suggested 50): ")
)
VERIFY_RAND_ITS = int(input("Enter the number of testing iterations (suggested 250): "))


def american_to_decimal(odds_int):
    if odds_int > 0:
        return 1 + (odds_int / 100)
    else:
        return 1 + (100 / abs(odds_int))


def get_market_strat_data(
    games_f,
    mom_tol,
    bet_streak_exp,
    win_strat,
    bet_against_mom,
    teams_f,
):

    team_d = dict()

    for t in teams_f.itertuples(index=False):
        games_suc = 0
        games_unsuc = 0
        num_games_waived = 0

        tid = t.tid
        team_d[tid] = dict()
        team_d[tid]["samp_rets"] = list()
        team_d[tid]["samp_money"] = list()
        team_d[tid]["rand_rets"] = list()
        team_d[tid]["rand_money"] = list()
        streak = 0

        # ensure chronological order
        tgames_f = games_f[
            (games_f["hid"] == tid) | (games_f["aid"] == tid)
        ].sort_values("unix")

        for g in tgames_f.itertuples(index=False):
            # skip games with no result
            if pd.isna(g.result_tid):
                continue

            # Determine if this team is home or away
            is_home = g.hid == tid
            team_odds = g.hodds if is_home else g.aodds
            opponent_odds = g.aodds if is_home else g.hodds

            # Skip games with missing odds
            if pd.isna(team_odds) or pd.isna(opponent_odds):
                streak = 0
                continue

            # Safe conversion and decimal odds
            try:
                team_odds_int = int(team_odds)
                opponent_odds_int = int(opponent_odds)
                team_dec = american_to_decimal(team_odds_int)
                opponent_dec = american_to_decimal(opponent_odds_int)
            except Exception:
                streak = 0
                continue

            # Determine betting strategy
            if win_strat:
                bet_on_team_to_win = True
            else:
                bet_on_team_to_win = False

            if bet_against_mom:
                bet_on_team_to_win = not bet_on_team_to_win

            # Bet only if PRIOR streak >= MOM_TOL (before seeing this game's result)
            if streak >= mom_tol:
                bet_amnt = (streak - mom_tol + 1) ** bet_streak_exp

                # Determine if our bet won and which odds we're using
                if bet_on_team_to_win:
                    bet_won = g.result_tid == tid
                    dec_odds = team_dec
                    odds_int = team_odds_int
                else:
                    bet_won = g.result_tid != tid
                    dec_odds = opponent_dec
                    odds_int = opponent_odds_int

                # Calculate strategy return
                if bet_won:
                    games_suc += 1
                    money_won = (dec_odds - 1) * bet_amnt
                else:
                    games_unsuc += 1
                    money_won = -bet_amnt

                # Per-bet return for strategy
                team_d[tid]["samp_rets"].append(money_won / bet_amnt)
                team_d[tid]["samp_money"].append(money_won)

                # Random baseline: bet $1 on random side (flat betting)
                if np.random.rand() < 0.5:
                    # Bet on home side
                    rand_dec_odds = american_to_decimal(g.hodds)
                    rand_bet_home = True
                else:
                    # Bet on away side
                    rand_dec_odds = american_to_decimal(g.aodds)
                    rand_bet_home = False

                # Use actual outcome
                rand_bet_won = (
                    (g.result_tid == g.hid)
                    if rand_bet_home
                    else (g.result_tid == g.aid)
                )

                if rand_bet_won:
                    rand_won = (rand_dec_odds - 1) * 1.0
                else:
                    rand_won = -1.0

                team_d[tid]["rand_rets"].append(rand_won / 1.0)
                team_d[tid]["rand_money"].append(rand_won)
            else:
                num_games_waived += 1

            # NOW determine if streak continues (after betting decision)
            if win_strat:
                streak_continues = g.result_tid == tid
            else:
                streak_continues = g.result_tid != tid

            # Update streak AFTER betting
            streak = streak + 1 if streak_continues else 0

        num_games = games_suc + games_unsuc
        team_d[tid]["suc_perc"] = (games_suc / num_games) if num_games > 0 else None
        team_d[tid]["num_games"] = num_games
        team_d[tid]["name"] = t.name
        team_d[tid]["num_games_waived"] = num_games_waived

    return team_d


def get_db(market, data_split):
    games_f = None
    teams_f = None
    with sqlite3.connect(DB_SLUG) as conn:
        games_f = pd.read_sql_query(
            f"""
                                    SELECT g.gid, g.hid, g.aid, g.result_tid, g.unix, g.hodds, g.aodds
                                    FROM games g
                                    WHERE g.mkt = '{market}'
                                    ORDER BY g.unix ASC
                                    """,
            conn,
        )
        teams_f = pd.read_sql_query(
            f"""
                                    SELECT t.tid, t.name
                                    FROM teams t
                                    WHERE t.mkt = '{market}'
                                    ORDER BY t.tid
                                    """,
            conn,
        )
    market_seed = abs(hash(market)) % (2**31)
    teams_f = teams_f.sample(frac=1.0, random_state=market_seed).reset_index(drop=True)
    len_train = int(len(teams_f) * (1 - VERIFY_PROP))
    teams_port = (
        slice(0, len_train) if data_split == "train" else slice(len_train, len(teams_f))
    )
    teams_f = teams_f.iloc[teams_port]

    return games_f, teams_f


def get_market_games_and_rets(team_d):
    # Collect all individual bet returns
    all_strat_rets = [ret for d in team_d.values() for ret in d["samp_rets"]]
    all_strat_money = [m for d in team_d.values() for m in d["samp_money"]]
    all_rand_rets = [ret for d in team_d.values() for ret in d["rand_rets"]]
    all_rand_money = [m for d in team_d.values() for m in d["rand_money"]]

    market_games = sum(
        d["num_games"] for d in team_d.values() if pd.notna(d["num_games"])
    )
    market_games_waived = sum(
        d["num_games_waived"]
        for d in team_d.values()
        if pd.notna(d["num_games_waived"])
    )

    # Average RETURN per bet (not dollar profit)
    market_return = np.mean(all_strat_rets) if all_strat_rets else 0.0
    market_ret_rand = np.mean(all_rand_rets) if all_rand_rets else 0.0

    return market_games, market_return, market_ret_rand


def permutation_test(strat_returns, rand_returns, n_permutations=10000, seed=None):
    """
    Permutation test to see if strategy returns are significantly better than random.
    Returns p-value for one-sided test (strat > random).
    """
    if seed is not None:
        np.random.seed(seed)

    observed_diff = np.mean(strat_returns) - np.mean(rand_returns)
    combined = np.concatenate([strat_returns, rand_returns])
    n_strat = len(strat_returns)

    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_strat = combined[:n_strat]
        perm_rand = combined[n_strat:]
        perm_diff = np.mean(perm_strat) - np.mean(perm_rand)
        if perm_diff >= observed_diff:
            count_extreme += 1

    return count_extreme / n_permutations


def get_train_results():
    train_results_d = dict()
    MARKETS = trim_markets()

    for market in MARKETS:
        param_scores_d = dict()
        games_f, teams_f = get_db(market, "train")

        for mom_tol in MOM_TOLS:
            for bet_streak_exp in BET_STREAK_EXPS:
                for win_strat in WIN_STRATS:
                    for bet_against_mom in BET_AGAINST_MOM:
                        # Fixed seed generation to avoid collisions
                        seed = hash(
                            (mom_tol, bet_streak_exp, win_strat, bet_against_mom)
                        ) % (2**31)
                        team_d = get_market_strat_data(
                            games_f,
                            mom_tol,
                            bet_streak_exp,
                            win_strat,
                            bet_against_mom,
                            teams_f,
                        )
                        all_samp_rets = (
                            np.concatenate(
                                [
                                    d["samp_rets"]
                                    for d in team_d.values()
                                    if d["samp_rets"]
                                ]
                            )
                            if any(d["samp_rets"] for d in team_d.values())
                            else np.array([])
                        )
                        all_rand_rets = (
                            np.concatenate(
                                [
                                    d["rand_rets"]
                                    for d in team_d.values()
                                    if d["rand_rets"]
                                ]
                            )
                            if any(d["rand_rets"] for d in team_d.values())
                            else np.array([])
                        )
                        if len(all_samp_rets) < (250 * (1 - VERIFY_PROP) / VERIFY_PROP):
                            score = -np.inf  # Penalize insufficient data
                        else:
                            diffs = all_samp_rets - all_rand_rets
                            score = np.mean(all_samp_rets)
                        param_scores_d[
                            (mom_tol, bet_streak_exp, win_strat, bet_against_mom)
                        ] = score

        train_results_d[market] = max(param_scores_d.items(), key=lambda x: x[1])[0]

    return train_results_d


results_d = dict()
train_res_d = get_train_results()

for market, params in train_res_d.items():
    mom_tol, bet_streak_exp, win_strat, bet_against_mom = params
    games_f, teams_f = get_db(market, "test")

    # Bootstrap test set
    market_return_list = []
    market_ret_rand_list = []
    tot_suc_list = []
    market_games_list = []
    tot_wins_list = []

    for iteration in range(VERIFY_RAND_ITS):
        teams_bootstrap = teams_f.sample(
            n=len(teams_f), replace=True, random_state=1000 + iteration
        )

        team_d = get_market_strat_data(
            games_f,
            mom_tol,
            bet_streak_exp,
            win_strat,
            bet_against_mom,
            teams_bootstrap,
        )
        market_games, market_return_it, market_ret_rand_it = get_market_games_and_rets(
            team_d
        )

        market_return_list.append(market_return_it)
        market_ret_rand_list.append(market_ret_rand_it)
        market_games_list.append(market_games)

        tot_wins = sum(
            sum(1 for ret in d["samp_rets"] if ret > 0) for d in team_d.values()
        )
        tot_bets = sum(len(d["samp_rets"]) for d in team_d.values())
        tot_suc_perc_it = (tot_wins / tot_bets) if tot_bets > 0 else 0.0
        tot_suc_list.append(tot_suc_perc_it)
        tot_wins_list.append(tot_wins)

    # Calculate averages across bootstrap iterations
    market_return = np.mean(market_return_list)
    market_ret_rand = np.mean(market_ret_rand_list)
    avg_win_rate = np.mean(tot_suc_list)
    market_games = int(np.mean(market_games_list))

    # Confidence interval
    ret_diffs = np.array(market_return_list) - np.array(market_ret_rand_list)
    ci_low = np.percentile(ret_diffs, 2.5)
    ci_high = np.percentile(ret_diffs, 97.5)

    # Collect all returns for significance test
    all_strat_rets = []
    all_rand_rets = []
    for iteration in range(min(100, VERIFY_RAND_ITS)):
        teams_bootstrap = teams_f.sample(
            n=len(teams_f), replace=True, random_state=1000 + iteration
        )
        team_d = get_market_strat_data(
            games_f,
            mom_tol,
            bet_streak_exp,
            win_strat,
            bet_against_mom,
            teams_bootstrap,
        )
        for d in team_d.values():
            all_strat_rets.extend(d["samp_rets"])
            all_rand_rets.extend(d["rand_rets"])

    if len(all_strat_rets) > 0:
        diffs = np.array(all_strat_rets) - np.array(all_rand_rets)
        res = stats.ttest_1samp(diffs, 0, alternative="greater")
        p_paired = res.pvalue
        p_perm = permutation_test(all_strat_rets, all_rand_rets, n_permutations=1000)
    else:
        p_paired = 1.0
        p_perm = 1.0

    market_d = dict()
    market_d["num_games"] = market_games
    market_d["avg_return_per_bet"] = market_return
    market_d["avg_return_rand_per_bet"] = market_ret_rand
    market_d["ret_diff"] = market_return - market_ret_rand
    market_d["tot_suc_perc"] = avg_win_rate
    market_d["tot_wins"] = np.mean(tot_wins_list)
    market_d["tot_suc"] = avg_win_rate * market_games
    market_d["p_val_paired_ttest"] = p_paired
    market_d["p_val_permutation"] = p_perm
    market_d["ret_diff_ci_lower"] = ci_low
    market_d["ret_diff_ci_upper"] = ci_high
    win_strat_str = ("Positive" if win_strat else "Negative") + " Momentum"
    market_d["win_strat_str"] = win_strat_str
    market_d["mom_tol"] = mom_tol
    market_d["bet_streak_exp"] = bet_streak_exp
    market_d["total_bets"] = market_games
    market_d["bet_against_mom"] = bet_against_mom

    OUT_PATH = "out/"
    PLOTS_PATH = f"{OUT_PATH}plots/"

    # Plot histogram
    Path(PLOTS_PATH).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    returns_pct = [r * 100 for r in market_return_list]
    plt.hist(returns_pct, bins=20, edgecolor="white", color="purple", alpha=0.7)
    plt.title(
        f"Bootstrap Distribution of Returns for {market.upper()}\n({VERIFY_RAND_ITS} bootstrap iterations)"
    )
    plt.xlabel("Average Return per Bet (%)")
    plt.ylabel("Frequency")
    plt.axvline(
        x=market_return * 100,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean Strategy = {market_return*100:.2f}%",
    )
    plt.axvline(
        x=market_ret_rand * 100,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Random Baseline = {market_ret_rand*100:.2f}%",
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(
        f"{PLOTS_PATH}{market}_bootstrap_returns.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    results_d[market] = market_d

with open(f"{OUT_PATH}results.json", "w") as o:
    json.dump(results_d, o, separators=(",", ": "), indent=3)
