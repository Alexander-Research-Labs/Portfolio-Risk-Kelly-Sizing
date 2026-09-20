import json

import config
import data


def prompt_entries(prompt_label):
    entries = []
    print(f"Enter {prompt_label}, one per line as 'TICKER SHARES'. Blank line to finish.")
    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 2:
            print("  format: TICKER SHARES, e.g. AAPL 40")
            continue
        ticker, shares = parts[0].upper(), parts[1]
        try:
            shares = float(shares)
        except ValueError:
            print("  shares must be a number")
            continue
        try:
            price = data.latest_price(ticker)
        except Exception as e:
            print(f"  could not fetch {ticker}: {e}")
            continue
        print(f"  {ticker}: {shares} shares @ ${price:,.2f} = ${shares * price:,.2f}")
        entries.append({"ticker": ticker, "shares": shares})
    return entries


def prompt_candidates():
    print("\nEnter candidate tickers you're considering adding, one per line. Blank line to finish.")
    tickers = []
    while True:
        line = input("> ").strip().upper()
        if not line:
            break
        ticker = line.split()[0]
        try:
            price = data.latest_price(ticker)
        except Exception as e:
            print(f"  could not fetch {ticker}: {e}")
            continue
        print(f"  {ticker}: ${price:,.2f}")
        tickers.append(ticker)
    return tickers


def main():
    holdings = prompt_entries("your current holdings")
    with open(config.HOLDINGS_PATH, "w") as f:
        json.dump(holdings, f, indent=1)
    print(f"\nWrote {len(holdings)} holdings to {config.HOLDINGS_PATH}")

    candidates = prompt_candidates()
    with open(config.CANDIDATES_PATH, "w") as f:
        json.dump(candidates, f, indent=1)
    print(f"Wrote {len(candidates)} candidates to {config.CANDIDATES_PATH}")

    all_tickers = sorted(set([h["ticker"] for h in holdings] + candidates))
    if all_tickers:
        print(f"\nCalibrating {len(all_tickers)} tickers: {', '.join(all_tickers)}")
        params = data.calibrate_universe(all_tickers)
        data.save_params(params)
        print(f"Saved calibration to {config.PARAMS_CACHE_PATH}")

    print("\nRun `python stress_test.py` for VaR/CVaR, or `python kelly.py TICKER` to size a candidate.")


if __name__ == "__main__":
    main()
