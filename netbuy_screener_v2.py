#!/usr/bin/env python3
"""기관 순매수 스크리너 v2.

단순 누적 순매수 상위가 아니라, "종목 자신의 역사 대비 이례적이고, 능동적 주체가,
꾸준히 나눠 사고 있으며, 가격이 아직 크게 반응하지 않은" 종목을 스코어링한다.

--------------------------------------------------------------------------
README NOTE — core_netbuy 정의와 공표치의 차이
--------------------------------------------------------------------------
이 스크립트가 사용하는 core_netbuy = 기관합계 순매수 − 금융투자 순매수 이다.

KRX/네이버 증권 등에서 통상 "기관 순매수"로 공표하는 수치는 기관합계 그대로이며,
그 안에는 금융투자(증권사 고유계정·ELS/ETF 헤지·프로그램매매 비중이 큰 주체)가
섞여 있다. 금융투자는 방향성 베팅보다 상품 헤지/차익거래 목적의 기계적 매매가
많아, "이 회사를 보고 능동적으로 사들이는 기관 자금"을 보려는 이 스크리너의
목적에는 잡음에 가깝다고 보고 의도적으로 제외했다.

따라서 이 스크립트의 core_netbuy, 그리고 이를 사용하는 zscore/ratio/score는
KRX 공식 통계나 네이버 증권의 "기관 순매수" 수치와 다르며, 직접 비교・검증에
쓰면 안 된다. (연기금/투신/사모 등 개별 주체 분해치는 공표치와 동일한 정의다.)
--------------------------------------------------------------------------

사용 예:
    python netbuy_screener_v2.py --date 20260703 --market ALL --top 30
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
import pandas as pd
from pykrx import stock
from tqdm import tqdm

import netbuy_cache as cache

CONFIG = {
    "LOOKBACK_DAYS": 250,
    "WINDOWS": {"short": 5, "long": 20},
    "MIN_HISTORY_FOR_Z": 100,  # 롤링합 분포의 유효 관측치가 이보다 적으면 zscore는 NaN
    "MIN_LISTED_DAYS": 120,
    "HARD_FILTERS": {
        "min_market_cap": 80_000_000_000,  # 800억
        "min_avg_trading_value": 500_000_000,  # 5억 (W20 일평균 거래대금)
        "min_persistence_w20": 0.55,
    },
    "QUADRANT": {"z_min": 1.0, "ret_quiet": 0.05},
    "WEIGHTS": {
        "zscore_w20": 0.40,
        "persistence_w20": 0.20,
        "accel": 0.15,
        "ratio_tval_w20": 0.15,
        "quiet_accumulation_bonus": 0.10,
    },
    "PENALTIES": {
        "block_deal": -20,
        "ft_share_high": -20,
    },
    "BLOCK_DEAL_MAX_DAY_CONTRIB": 0.4,
    "FT_SHARE_THRESHOLD": 0.6,
    "ACCEL_CLIP": (1.0, 2.0),
    "RATIO_TVAL_CAP": 0.15,  # ratio_tval_w20 정규화 상한 (경험적 캡, 필요시 조정)
    "INVESTORS": ["기관합계", "금융투자", "투신", "연기금", "사모"],
    "RETRY_SLEEP": 0.3,
    "RETRY_COUNT": 1,
    "CACHE_DIR": cache.DEFAULT_CACHE_DIR,
}


# ---------------------------------------------------------------------------
# 거래일 / 유니버스
# ---------------------------------------------------------------------------

def resolve_base_date(date: str | None) -> str:
    if date is None:
        date = pd.Timestamp.today().strftime("%Y%m%d")
    return stock.get_nearest_business_day_in_a_week(date, prev=True)


def get_trading_days(base_date: str, lookback_days: int) -> list[str]:
    calendar_from = (
        pd.Timestamp(base_date) - pd.Timedelta(days=int(lookback_days * 1.6) + 30)
    ).strftime("%Y%m%d")
    days = stock.get_previous_business_days(fromdate=calendar_from, todate=base_date)
    days = [d.strftime("%Y%m%d") for d in days]
    if len(days) < lookback_days:
        raise RuntimeError(
            f"거래일 조회 실패: {len(days)}일만 확보됨 (필요: {lookback_days}일). "
            "calendar_from을 더 과거로 늘려야 할 수 있습니다."
        )
    return days[-lookback_days:]


def _is_preferred_stock(ticker: str) -> bool:
    """KRX 관행상 보통주 티커는 '0'으로 끝난다. 우선주는 대체로 그 외 숫자로
    끝나므로 이를 근사 판별에 사용한다 (완전하지 않은 휴리스틱)."""
    return ticker[-1] != "0"


def build_universe_filter(base_date: str, market: str) -> tuple[set[str], dict[str, str]]:
    """ETF/ETN/우선주를 제외한 후보 티커 집합과 ticker->market 매핑을 반환한다.
    스팩/리츠는 종목명이 필요해 collect_history 이후 단계에서 추가로 제외한다."""
    markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]
    ticker_market: dict[str, str] = {}
    for m in markets:
        for t in stock.get_market_ticker_list(base_date, market=m):
            ticker_market[t] = m

    excluded = set(stock.get_etf_ticker_list(base_date)) | set(stock.get_etn_ticker_list(base_date))
    valid = {t for t in ticker_market if t not in excluded and not _is_preferred_stock(t)}
    return valid, ticker_market


# ---------------------------------------------------------------------------
# 데이터 수집 (캐시 우선)
# ---------------------------------------------------------------------------

def _fetch_with_retry(fn: Callable, *args, retries: int = CONFIG["RETRY_COUNT"],
                       sleep: float = CONFIG["RETRY_SLEEP"]):
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = fn(*args)
            time.sleep(sleep)
            return result
        except Exception as e:  # pykrx는 다양한 예외를 던지므로 넓게 잡고 재시도
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"{getattr(fn, '__name__', fn)}{args} 호출 실패: {last_err}")


def _fetch_investor_day(date: str, market: str, investor: str) -> pd.DataFrame:
    raw = stock.get_market_net_purchases_of_equities(date, date, market, investor)
    if raw.empty:
        return pd.DataFrame(columns=["종목명", "순매수거래대금"])
    return raw[["종목명", "순매수거래대금"]]


def _fetch_snapshot_day(date: str, market: str) -> pd.DataFrame:
    raw = stock.get_market_cap_by_ticker(date, market=market)
    return raw[["종가", "시가총액", "거래대금"]]


def collect_history(
    base_date: str, market: str, lookback_days: int, cache_dir: str, rebuild_cache: bool
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    trading_days = get_trading_days(base_date, lookback_days)
    investors = CONFIG["INVESTORS"]

    if rebuild_cache:
        n = cache.clear(cache_dir, market)
        print(f"캐시 삭제: {n}개 파일 (market={market})")

    netbuy_frames: dict[str, dict[str, pd.Series]] = {inv: {} for inv in investors}
    close_frames: dict[str, pd.Series] = {}
    tval_frames: dict[str, pd.Series] = {}
    mcap_frames: dict[str, pd.Series] = {}
    name_map: dict[str, str] = {}

    for date in tqdm(trading_days, desc=f"[{market}] 일자별 데이터 수집"):
        for inv in investors:
            df = cache.load(cache_dir, date, market, inv)
            if df is None:
                df = _fetch_with_retry(_fetch_investor_day, date, market, inv)
                cache.save(cache_dir, date, market, inv, df)
            netbuy_frames[inv][date] = df["순매수거래대금"].astype(float)
            name_map.update(df["종목명"].to_dict())

        snap = cache.load(cache_dir, date, market, "시세")
        if snap is None:
            snap = _fetch_with_retry(_fetch_snapshot_day, date, market)
            cache.save(cache_dir, date, market, "시세", snap)
        close_frames[date] = snap["종가"].astype(float)
        tval_frames[date] = snap["거래대금"].astype(float)
        mcap_frames[date] = snap["시가총액"].astype(float)

    def _to_wide(frames: dict[str, pd.Series]) -> pd.DataFrame:
        wide = pd.DataFrame(frames).T
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()

    netbuy_wide = {inv: _to_wide(netbuy_frames[inv]) for inv in investors}
    close_wide = _to_wide(close_frames)
    tval_wide = _to_wide(tval_frames)
    mcap_wide = _to_wide(mcap_frames)

    return netbuy_wide, close_wide, tval_wide, mcap_wide, name_map


# ---------------------------------------------------------------------------
# 파생 신호
# ---------------------------------------------------------------------------

def compute_core_netbuy(netbuy_wide: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inst, prop = netbuy_wide["기관합계"].align(netbuy_wide["금융투자"], join="outer", axis=1, fill_value=0.0)
    inst = inst.fillna(0.0)
    prop = prop.fillna(0.0)
    return inst - prop, inst, prop


def rolling_zscore(core: pd.DataFrame, window: int, min_obs: int) -> pd.Series:
    rolling_sum = core.rolling(window=window, min_periods=window).sum()
    mean = rolling_sum.mean()
    std = rolling_sum.std()
    n_obs = rolling_sum.notna().sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (rolling_sum.iloc[-1] - mean) / std
    z[(n_obs < min_obs) | (std == 0)] = np.nan
    return z


def compute_persistence(core_window: pd.DataFrame) -> pd.Series:
    valid_days = core_window.notna().sum()
    pos_days = (core_window > 0).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return pos_days / valid_days.replace(0, np.nan)


def compute_max_day_contrib(core_window: pd.DataFrame) -> pd.Series:
    total = core_window.sum()
    max_day = core_window.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        contrib = max_day / total
    contrib[total <= 0] = np.nan
    return contrib


def compute_streak(core: pd.DataFrame) -> pd.Series:
    """기준일(마지막 행)에서 역산해 연속으로 core > 0 인 거래일 수."""
    arr = core.to_numpy()
    n_days, n_tickers = arr.shape
    streak = np.zeros(n_tickers, dtype=int)
    for j in range(n_tickers):
        count = 0
        for i in range(n_days - 1, -1, -1):
            v = arr[i, j]
            if not (v > 0):  # NaN, 0, 음수에서 모두 중단
                break
            count += 1
        streak[j] = count
    return pd.Series(streak, index=core.columns)


def compute_accel(core: pd.DataFrame, w_short: int, w_long: int) -> pd.Series:
    avg_short = core.iloc[-w_short:].mean()
    avg_long = core.iloc[-w_long:].mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = avg_short / avg_long
    accel[avg_long <= 0] = np.nan
    return accel


def compute_ft_share(core_sum: pd.Series, prop_sum: pd.Series) -> pd.Series:
    denom = core_sum.abs() + prop_sum.abs()
    with np.errstate(invalid="ignore", divide="ignore"):
        share = prop_sum.abs() / denom
    share[denom == 0] = np.nan
    return share


def compute_ret_w20(close_wide: pd.DataFrame, window: int) -> pd.Series:
    if len(close_wide) <= window:
        return pd.Series(np.nan, index=close_wide.columns)
    now = close_wide.iloc[-1]
    then = close_wide.iloc[-1 - window]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = now / then - 1.0
    ret[then <= 0] = np.nan
    return ret


def classify_quadrant(z: pd.Series, ret: pd.Series, z_min: float, ret_quiet: float) -> pd.Series:
    passed = z >= z_min
    out = pd.Series("미통과", index=z.index)
    out[passed & (ret >= ret_quiet)] = "추세 동반"
    out[passed & (ret < ret_quiet)] = "조용한 축적"
    out[z.isna() | ret.isna()] = "미통과"
    return out


def compute_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    w = cfg["WEIGHTS"]

    z_norm = ((df["zscore_w20"].clip(lower=0.0, upper=3.0)) / 3.0).fillna(0.0)

    a_lo, a_hi = cfg["ACCEL_CLIP"]
    accel_norm = ((df["accel"].clip(lower=a_lo, upper=a_hi) - a_lo) / (a_hi - a_lo)).fillna(0.0)

    tval_cap = cfg["RATIO_TVAL_CAP"]
    tval_norm = (df["ratio_tval_w20"].clip(lower=0.0, upper=tval_cap) / tval_cap).fillna(0.0)

    persistence_norm = df["persistence_w20"].fillna(0.0)
    quiet_bonus = (df["quadrant"] == "조용한 축적").astype(float)

    base = (
        w["zscore_w20"] * z_norm
        + w["persistence_w20"] * persistence_norm
        + w["accel"] * accel_norm
        + w["ratio_tval_w20"] * tval_norm
        + w["quiet_accumulation_bonus"] * quiet_bonus
    ) * 100.0

    penalty = np.where(df["block_deal_flag"], cfg["PENALTIES"]["block_deal"], 0.0)
    penalty = penalty + np.where(df["ft_share"] > cfg["FT_SHARE_THRESHOLD"], cfg["PENALTIES"]["ft_share_high"], 0.0)

    return (base + penalty).clip(lower=0.0, upper=100.0)


# ---------------------------------------------------------------------------
# 검증용 리포트
# ---------------------------------------------------------------------------

def print_distribution_report(df: pd.DataFrame, cfg: dict) -> None:
    print("\n=== [검증 1] 분포 리포트 (하드필터 적용 전, 전체 유니버스 기준) ===")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for col in ["zscore_w5", "zscore_w20", "ratio_tval_w20", "persistence_w20"]:
        series = df[col].dropna()
        if series.empty:
            print(f"{col}: 유효 데이터 없음")
            continue
        pct = np.percentile(series, percentiles)
        line = ", ".join(f"p{p}={v:.3f}" for p, v in zip(percentiles, pct))
        print(f"{col} (n={len(series)}): {line}")

    z_min = cfg["QUADRANT"]["z_min"]
    ret_quiet = cfg["QUADRANT"]["ret_quiet"]
    min_persist = cfg["HARD_FILTERS"]["min_persistence_w20"]
    n_total = len(df)
    if n_total == 0:
        print("유니버스가 비어 있습니다.")
        return
    n_zpass = int((df["zscore_w20"] >= z_min).sum())
    n_quiet = int((df["quadrant"] == "조용한 축적").sum())
    n_trend = int((df["quadrant"] == "추세 동반").sum())
    n_persist = int((df["persistence_w20"] >= min_persist).sum())
    print(f"\nzscore_w20 >= {z_min} 통과: {n_zpass}/{n_total} ({n_zpass/n_total:.1%})")
    print(f"quadrant='조용한 축적': {n_quiet}/{n_total} ({n_quiet/n_total:.1%})")
    print(f"quadrant='추세 동반' (ret_w20>={ret_quiet:.0%}): {n_trend}/{n_total} ({n_trend/n_total:.1%})")
    print(f"persistence_w20 >= {min_persist}: {n_persist}/{n_total} ({n_persist/n_total:.1%})")


def pick_default_samples(mcap_now: pd.Series, n: int = 3) -> list[str]:
    ranked = mcap_now.dropna().sort_values(ascending=False)
    if len(ranked) == 0:
        return []
    if len(ranked) < n:
        return list(ranked.index)
    idx = [0, len(ranked) // 2, len(ranked) - 1]  # 대형/중형/소형
    return [ranked.index[i] for i in idx]


def validate_samples(
    core: pd.DataFrame, inst: pd.DataFrame, prop: pd.DataFrame,
    tickers: list[str], name_map: dict[str, str], window: int,
) -> None:
    print("\n=== [검증 2] 샘플 종목 원자료 대조 (대형/중형/소형) ===")
    for t in tickers:
        if t not in core.columns:
            print(f"\n{t}: 유니버스에 없음 (하드필터 이전 제외 또는 데이터 없음)")
            continue
        detail = pd.DataFrame({
            "기관합계": inst[t].iloc[-window:],
            "금융투자": prop[t].iloc[-window:],
            "core(직접차감)": (inst[t] - prop[t]).iloc[-window:],
            "core(파이프라인계산)": core[t].iloc[-window:],
        })
        name = name_map.get(t, t)
        print(f"\n--- {name} ({t}) 최근 {window}거래일 ---")
        print(detail.to_string())
        match = np.allclose(detail["core(직접차감)"], detail["core(파이프라인계산)"], equal_nan=True)
        print(f"core_netbuy 재계산 일치 여부: {match}")
        print(f"Σcore={detail['core(파이프라인계산)'].sum():,.0f}원, "
              f"persistence={(detail['core(파이프라인계산)'] > 0).mean():.2f}")


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def run_screener(args: argparse.Namespace):
    cfg = CONFIG
    base_date = resolve_base_date(args.date)
    print(f"기준일: {base_date} (시장: {args.market})")

    valid_tickers, ticker_market = build_universe_filter(base_date, args.market)

    netbuy_wide, close_wide, tval_wide, mcap_wide, name_map = collect_history(
        base_date, args.market, cfg["LOOKBACK_DAYS"], args.cache_dir, args.rebuild_cache
    )

    core, inst, prop = compute_core_netbuy(netbuy_wide)

    candidate_cols = [t for t in valid_tickers if t in mcap_wide.columns]
    missing_names = [t for t in candidate_cols if t not in name_map]
    for t in missing_names:
        try:
            name_map[t] = stock.get_market_ticker_name(t)
        except Exception:
            name_map[t] = t

    name_series = pd.Series({t: name_map.get(t, t) for t in candidate_cols})
    is_spac_or_reit = name_series.str.contains("스팩", na=False) | name_series.str.contains("리츠", na=False)
    universe_cols = [t for t in candidate_cols if not is_spac_or_reit.get(t, False)]

    core = core.reindex(columns=universe_cols, fill_value=0.0)
    inst = inst.reindex(columns=universe_cols, fill_value=0.0)
    prop = prop.reindex(columns=universe_cols, fill_value=0.0)
    close_wide = close_wide.reindex(columns=universe_cols)
    tval_wide = tval_wide.reindex(columns=universe_cols)
    mcap_wide = mcap_wide.reindex(columns=universe_cols)

    W = cfg["WINDOWS"]
    core_w5 = core.iloc[-W["short"]:]
    core_w20 = core.iloc[-W["long"]:]
    prop_w20 = prop.iloc[-W["long"]:]

    sum_w5 = core_w5.sum()
    sum_w20 = core_w20.sum()

    z_w5 = rolling_zscore(core, W["short"], cfg["MIN_HISTORY_FOR_Z"])
    z_w20 = rolling_zscore(core, W["long"], cfg["MIN_HISTORY_FOR_Z"])
    persistence_w20 = compute_persistence(core_w20)
    max_day_contrib = compute_max_day_contrib(core_w20)
    streak = compute_streak(core)
    accel = compute_accel(core, W["short"], W["long"])
    ret_w20 = compute_ret_w20(close_wide, W["long"])
    ft_share = compute_ft_share(sum_w20, prop_w20.sum())

    mcap_now = mcap_wide.iloc[-1]
    tval_w20 = tval_wide.iloc[-W["long"]:]
    tval_avg_w20 = tval_w20.mean()
    listed_days = close_wide.notna().sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_mcap_w20 = sum_w20 / mcap_now.replace(0, np.nan)
        ratio_tval_w20 = sum_w20 / tval_w20.sum().replace(0, np.nan)

    quadrant = classify_quadrant(z_w20, ret_w20, cfg["QUADRANT"]["z_min"], cfg["QUADRANT"]["ret_quiet"])

    df = pd.DataFrame({
        "종목": [name_map.get(t, t) for t in universe_cols],
        "시장": [ticker_market.get(t, args.market) for t in universe_cols],
        "시가총액(억)": (mcap_now / 1e8).round(1),
        "core_w5(억)": (sum_w5 / 1e8).round(2),
        "core_w20(억)": (sum_w20 / 1e8).round(2),
        "zscore_w5": z_w5.round(2),
        "zscore_w20": z_w20.round(2),
        "persistence_w20": persistence_w20.round(3),
        "accel": accel.round(2),
        "ret_w20": ret_w20.round(4),
        "quadrant": quadrant,
        "ft_share": ft_share.round(3),
        "streak": streak,
        "max_day_contrib": max_day_contrib.round(3),
        "ratio_mcap_w20": ratio_mcap_w20.round(4),
        "ratio_tval_w20": ratio_tval_w20.round(4),
        "listed_days": listed_days,
        "tval_avg_w20(억)": (tval_avg_w20 / 1e8).round(2),
    }, index=pd.Index(universe_cols, name="티커"))

    df["block_deal_flag"] = df["max_day_contrib"] > cfg["BLOCK_DEAL_MAX_DAY_CONTRIB"]
    df["score"] = compute_score(df, cfg).round(1)

    def _flags(row) -> str:
        f = []
        if row["block_deal_flag"]:
            f.append("BLOCK_DEAL")
        if row["ft_share"] > cfg["FT_SHARE_THRESHOLD"]:
            f.append("FT_HEAVY")
        return ",".join(f)

    df["flags"] = df.apply(_flags, axis=1)

    # 주체별 분해 (연기금/투신/사모)
    breakdown = {}
    for inv in ["연기금", "투신", "사모"]:
        inv_w20 = netbuy_wide[inv].reindex(columns=universe_cols, fill_value=0.0).iloc[-W["long"]:].sum()
        breakdown[f"{inv}_w20(억)"] = (inv_w20 / 1e8).round(2)
    breakdown_df = pd.DataFrame(breakdown, index=pd.Index(universe_cols, name="티커"))
    breakdown_df.insert(0, "시장", df["시장"])
    breakdown_df.insert(0, "종목", df["종목"])
    breakdown_df["연기금_단독매수"] = (
        (breakdown_df["연기금_w20(억)"] > 0)
        & (breakdown_df["투신_w20(억)"] <= 0)
        & (breakdown_df["사모_w20(억)"] <= 0)
    )

    hard_pass = (
        (df["시가총액(억)"] * 1e8 >= cfg["HARD_FILTERS"]["min_market_cap"])
        & (df["tval_avg_w20(억)"] * 1e8 >= cfg["HARD_FILTERS"]["min_avg_trading_value"])
        & (df["listed_days"] >= cfg["MIN_LISTED_DAYS"])
        & (df["persistence_w20"] >= cfg["HARD_FILTERS"]["min_persistence_w20"])
    )
    result = df[hard_pass].sort_values("score", ascending=False)

    context = {
        "core": core, "inst": inst, "prop": prop,
        "mcap_now": mcap_now, "name_map": name_map,
        "base_date": base_date,
    }
    return df, result, breakdown_df, context


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="기관 순매수 스크리너 v2")
    p.add_argument("--date", default=None, help="기준일 YYYYMMDD (기본: 오늘, 휴장일이면 직전 거래일)")
    p.add_argument("--market", default="ALL", choices=["KOSPI", "KOSDAQ", "ALL"])
    p.add_argument("--out", default=None, help="CSV 출력 경로 (기본: netbuy_screener_{date}_{market}.csv)")
    p.add_argument("--min-score", type=float, default=60.0)
    p.add_argument("--rebuild-cache", action="store_true", help="캐시를 무시하고 전체 재수집")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--cache-dir", default=CONFIG["CACHE_DIR"])
    p.add_argument("--validate-samples", default=None, help="검증용 티커 콤마구분 (예: 005930,000660,123456)")
    p.add_argument("--skip-validation", action="store_true", help="분포 리포트/샘플 대조 생략")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df, result, breakdown_df, ctx = run_screener(args)

    print_distribution_report(df, CONFIG)

    if not args.skip_validation:
        sample_tickers = (
            [t.strip() for t in args.validate_samples.split(",")]
            if args.validate_samples
            else pick_default_samples(ctx["mcap_now"])
        )
        validate_samples(
            ctx["core"], ctx["inst"], ctx["prop"], sample_tickers, ctx["name_map"], CONFIG["WINDOWS"]["long"]
        )

    out_path = args.out or f"netbuy_screener_{ctx['base_date']}_{args.market}.csv"
    df.sort_values("score", ascending=False).to_csv(out_path, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {out_path} (전체 {len(df)}종목, 하드필터 통과 {len(result)}종목)")

    breakdown_path = out_path.rsplit(".", 1)[0] + "_주체별분해.csv"
    breakdown_df.to_csv(breakdown_path, encoding="utf-8-sig")
    print(f"주체별 분해 CSV 저장 완료: {breakdown_path}")

    top = result[result["score"] >= args.min_score].head(args.top)
    cols = [
        "종목", "시장", "시가총액(억)", "core_w5(억)", "core_w20(억)", "zscore_w5", "zscore_w20",
        "persistence_w20", "accel", "ret_w20", "quadrant", "ft_share", "flags", "score",
    ]
    print(f"\n=== 상위 {args.top} (하드필터 통과, score >= {args.min_score}) ===")
    if top.empty:
        print("조건을 만족하는 종목이 없습니다.")
    else:
        print(top[cols].to_string())


if __name__ == "__main__":
    main()
