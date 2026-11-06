"""A heterogeneous-agent consumption economy.

Each household belongs to one of four demographic archetypes. Archetypes differ in job-loss
exposure, rate sensitivity (mortgage and credit-card debt service), the liquid buffer they try to
hold, and how fast they spend down or rebuild that buffer. Every quarter, each household:

1. sees its permanent income grow with aggregate real disposable income,
2. becomes employed or unemployed with odds tied to the unemployment rate and its archetype,
3. earns labour income (or unemployment insurance at the replacement rate) plus interest,
4. sets a target buffer that rises with job-loss fear, gloomy sentiment and real rates,
5. consumes its permanent income plus a share of its cash above (or below) that target,
   subject to a borrowing limit, so liquidity-constrained households end up hand-to-mouth.

Aggregate consumption is the sum over households. Five behavioural parameters are shared across
archetypes (scaled by fixed archetype multipliers) and calibrated to 1980-2009 data.
"""
from dataclasses import dataclass

import numpy as np

ARCHETYPES = ['young professionals', 'families', 'students', 'retirees']
SHARE = np.array([0.30, 0.35, 0.10, 0.25])
INCOME = np.array([1.1, 1.3, 0.35, 0.8])      # relative permanent income
JOB_BETA = np.array([1.1, 0.9, 1.9, 0.08])    # unemployment exposure vs the headline rate
RATE_SENS = np.array([1.0, 1.6, 0.4, 0.3])    # debt-service / intertemporal substitution
BUFFER = np.array([0.9, 1.0, 0.25, 4.0])      # target liquid wealth, quarters of income
SPEED = np.array([1.0, 0.9, 1.6, 0.45])       # how fast excess cash is spent or rebuilt
BORROW = np.array([0.6, 0.8, 0.2, 0.3])       # credit limit, quarters of income

PARAMS = ['kappa', 'fear_u', 'fear_sent', 'rate', 'replace', 'smooth']
BOUNDS = [(0.02, 0.6), (0.0, 25.0), (0.0, 1.5), (0.0, 8.0), (0.2, 0.8), (0.05, 1.0)]
TREND = 0.55  # long-run real income growth, % per quarter (1980-2009 average)


@dataclass
class State:
    perm: np.ndarray
    wealth: np.ndarray
    employed: np.ndarray
    belief: float = TREND  # perceived trend income growth, % per quarter


class Economy:
    def __init__(self, n=4000, horizon=200, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.k = rng.choice(4, n, p=SHARE)
        self.perm0 = INCOME[self.k] * rng.lognormal(-0.08, 0.4, n)
        self.wealth0 = BUFFER[self.k] * self.perm0 * rng.lognormal(-0.125, 0.5, n)
        # common random numbers: the same idiosyncratic draws for every parameter vector,
        # so the calibration objective is smooth in the parameters
        self.keep = rng.random((horizon, n))
        self.draw = rng.random((horizon, n))

    def init(self):
        return State(self.perm0.copy(), self.wealth0.copy(), np.ones(self.n, bool))

    def step(self, s, d, theta, t):
        """Advance one quarter under drivers d = (g_y, u, rr, sent). Returns (state, consumption)."""
        kappa, fear_u, fear_sent, rate, replace, smooth = theta
        g_y, u, rr, sent = d
        k = self.k
        # households update their view of permanent income slowly (adaptive expectations);
        # the gap between realised and perceived income flows into the cash buffer instead
        belief = (1 - smooth) * s.belief + smooth * g_y
        perm = s.perm * np.exp(belief / 100)
        earned = s.perm * np.exp(g_y / 100)
        u_k = np.clip(u / 100 * JOB_BETA[k], 0, 0.6)
        employed = np.where(self.keep[t] < 0.7, s.employed, self.draw[t] >= u_k)
        income = earned * np.where(employed, 1.0, replace)
        wealth = s.wealth * (1 + rr / 400)
        fear = fear_u * u_k ** 2 + fear_sent * np.maximum(0, (85 - sent) / 25)
        target = BUFFER[k] * perm * (1 + fear + rate * RATE_SENS[k] * rr / 100)
        cons = perm + kappa * SPEED[k] * (wealth + income - perm - target)
        cons = np.clip(cons, 0.25 * perm, wealth + income + BORROW[k] * perm)
        return State(perm, wealth + income - cons, employed, belief), cons


def simulate(econ, drivers, forecast_drivers, theta):
    """Filter through realised drivers and make a one-step-ahead forecast every quarter.

    drivers[t] are realised at t; forecast_drivers[t] are the forecasts of t+1 made at t.
    Returns realised aggregate consumption C[t], the forecast growth for t+1 made at t, and
    consumption by archetype.
    """
    T = len(drivers)
    s = econ.init()
    # burn in on the first quarter so buffers settle around their targets
    for _ in range(12):
        s, _ = econ.step(s, (0.5, drivers[0][1], drivers[0][2], drivers[0][3]), theta, 0)
    C = np.zeros(T)
    by_k = np.zeros((T, 4))
    fc = np.full(T, np.nan)
    fc_k = np.full((T, 4), np.nan)
    for t in range(T):
        s, c = econ.step(s, drivers[t], theta, t)
        C[t] = c.sum()
        by_k[t] = np.bincount(econ.k, c, 4)
        if t + 1 < T:
            _, c_next = econ.step(s, forecast_drivers[t], theta, t + 1)
            fc[t] = np.log(c_next.sum() / C[t]) * 100
            fc_k[t] = (np.bincount(econ.k, c_next, 4) - by_k[t]) / C[t] * 100
    return C, fc, by_k, fc_k


def mpc(econ, s, d, theta, t, bump=0.05):
    """Marginal propensity to consume out of a one-off cash transfer, per household."""
    _, base = econ.step(s, d, theta, t)
    rich = State(s.perm, s.wealth + bump * s.perm, s.employed, s.belief)
    _, more = econ.step(rich, d, theta, t)
    return (more - base) / (bump * s.perm)
