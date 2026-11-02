"""Black-Scholes pricing for European options (no external dependencies)."""
import math


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    S      : spot price
    K      : strike
    T      : time to expiry in years
    r      : risk-free rate (continuous, e.g. 0.05 for 5%)
    sigma  : volatility (annualised, e.g. 0.2 for 20%)
    q      : continuous dividend yield
    Returns a dict with price and greeks.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return {"price": intrinsic}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    gamma = disc_q * norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * disc_q * norm_pdf(d1) * sqrtT
    theta_common = -S * disc_q * norm_pdf(d1) * sigma / (2 * sqrtT)

    if option_type == "call":
        price = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
        delta = disc_q * norm_cdf(d1)
        theta = theta_common - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        price = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)
        delta = -disc_q * norm_cdf(-d1)
        theta = theta_common + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
        rho = -K * T * disc_r * norm_cdf(-d2)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,     # per 1 vol point
        "theta": theta / 365,   # per calendar day
        "rho": rho / 100,       # per 1% change in rates
    }


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    for kind in ("call", "put"):
        res = black_scholes(S, K, T, r, sigma, kind)
        print(f"{kind.upper():4s}  " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))
