import json
import os
import sys
from datetime import datetime, timezone

import requests

import config
import data
import kelly
import stress_test

SUPABASE_URL = "https://ozucmxdtorvuaeqwuphf.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_368DneOW53RiYiP-DBGw5A_jDW-MeRD"
PUBLISH_ENDPOINT = "https://alexanderresearchlabs.com/api/qn8xzrpm4v"


def load_env():
    path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()


def get_access_token(email, password):
    res = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers={"apikey": SUPABASE_ANON_KEY, "content-type": "application/json"},
        json={"email": email, "password": password},
        timeout=15,
    )
    res.raise_for_status()
    return res.json()["access_token"]


def build_snapshot():
    holdings = stress_test.load_holdings()
    weights, prices, total_value = stress_test.portfolio_weights(holdings)
    params = data.load_params()

    portfolio_risk = {
        "10_day": stress_test.run_stress_test(weights, params, horizon_days=config.HORIZON_DAYS),
        "1_year": stress_test.run_stress_test(weights, params, horizon_days=config.HORIZON_DAYS_LONG),
    }

    try:
        with open(config.CANDIDATES_PATH) as f:
            candidates = json.load(f)
    except FileNotFoundError:
        candidates = []

    candidate_results = []
    for ticker in candidates:
        try:
            result = kelly.run(ticker)
            candidate_results.append({"ticker": ticker.upper(), **result})
        except Exception as e:
            candidate_results.append({"ticker": ticker.upper(), "error": str(e)})

    return {
        "holdings": holdings,
        "weights": weights,
        "prices": prices,
        "total_value": total_value,
        "portfolio_risk": portfolio_risk,
        "candidates": candidate_results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def publish_headers():
    secret = os.environ.get("ALR_PUBLISH_SECRET")
    if secret:
        return {"X-Publish-Secret": secret, "content-type": "application/json"}

    email = os.environ.get("ALR_ADMIN_EMAIL")
    password = os.environ.get("ALR_ADMIN_PASSWORD")
    if not email or not password:
        print("Set ALR_PUBLISH_SECRET in .env (preferred), or ALR_ADMIN_EMAIL and ALR_ADMIN_PASSWORD.")
        sys.exit(1)
    print("Signing in with the admin account (set ALR_PUBLISH_SECRET to avoid this)...")
    token = get_access_token(email, password)
    return {"Authorization": f"Bearer {token}", "content-type": "application/json"}


def main():
    load_env()

    print("Running the full simulation for the portfolio and every candidate...")
    snapshot = build_snapshot()

    headers = publish_headers()

    print("Publishing to Alexander Research Labs...")
    res = requests.post(
        PUBLISH_ENDPOINT,
        headers=headers,
        json=snapshot,
        timeout=30,
    )
    res.raise_for_status()
    print("Published:", res.json())


if __name__ == "__main__":
    main()
