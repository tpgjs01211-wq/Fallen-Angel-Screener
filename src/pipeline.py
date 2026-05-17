"""
Fallen Angel Daily Pipeline  ·  GitHub Actions 배포용
매일 오전 7시 KST (22:00 UTC) GitHub Actions 에서 실행
결과: docs/index.html  +  data/analysis_YYYYMMDD.csv
"""
import os, glob
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ── 경로 설정 (GitHub Actions: repo root 기준) ──────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR   = os.path.join(ROOT, "docs")
DATA_DIR   = os.path.join(ROOT, "data")

# ── 전략 파라미터 ────────────────────────────────────────
TOTAL_CAPITAL      = 50_000
MAX_POSITION_PCT   = 0.10
RISK_PER_TRADE     = 0.02

MIN_ROE            = 0.12
MAX_DEBT_TO_EQUITY = 150
MIN_GROSS_MARGIN   = 0.30
MIN_MARKET_CAP     = 5_000_000_000

DROP_THRESHOLD        = -0.15
DROP_LOOKBACK_DAYS    = 90
MIN_DROP_DAYS         = 5
REQUIRE_STABILIZATION = True
ANALYZE_TOP_N         = 10


# ════════════════════════════════════════════════════════
#  PHASE 1 : SCREENING
# ════════════════════════════════════════════════════════

def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0 (compatible; QuantScreener/1.0)"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=20
    )
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))[0]["Symbol"] \
             .str.replace(".", "-", regex=False).tolist()


def screen_quality(tickers):
    print("  [1/5] Quality screening …")
    quality = []
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(tickers)}")
        try:
            info = yf.Ticker(ticker).info
            if not info or not info.get("regularMarketPrice"):
                continue
            roe  = info.get("returnOnEquity")
            dte  = info.get("debtToEquity")
            gm   = info.get("grossMargins")
            mcap = info.get("marketCap", 0)
            if roe  is None or roe  < MIN_ROE:               continue
            if dte  is not None and dte  > MAX_DEBT_TO_EQUITY: continue
            if gm   is not None and gm   < MIN_GROSS_MARGIN:   continue
            if mcap < MIN_MARKET_CAP:                          continue
            quality.append({
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "marketCap": mcap,
                "roe": roe, "debtToEquity": dte, "grossMargins": gm,
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "earningsGrowth": info.get("earningsGrowth"),
                "revenueGrowth": info.get("revenueGrowth"),
                "currentRatio": info.get("currentRatio"),
                "freeCashflow": info.get("freeCashflow"),
                "dividendYield": info.get("dividendYield"),
            })
        except Exception:
            continue
    print(f"    Quality pass: {len(quality)}/{len(tickers)}")
    return pd.DataFrame(quality)


def screen_drops(quality_df):
    print("  [2/5] Drop screening …")
    tickers = quality_df["ticker"].tolist()
    start   = (datetime.now() - timedelta(days=DROP_LOOKBACK_DAYS + 30)).strftime("%Y-%m-%d")
    prices  = yf.download(tickers, start=start, progress=False)
    close   = prices["Close"] if isinstance(prices.columns, pd.MultiIndex) else prices[["Close"]]

    fallen = []
    for ticker in tickers:
        try:
            if ticker not in close.columns: continue
            series = close[ticker].dropna()
            if len(series) < DROP_LOOKBACK_DAYS // 2: continue

            recent    = series.tail(DROP_LOOKBACK_DAYS)
            peak      = float(recent.max())
            peak_date = recent.idxmax()
            current   = float(series.iloc[-1])
            drop_pct  = (current - peak) / peak
            if drop_pct > DROP_THRESHOLD: continue

            days_since = len(series[series.index > peak_date])
            if days_since < MIN_DROP_DAYS: continue

            post_peak = series[series.index >= peak_date]
            trough    = float(post_peak.min())
            bounce    = (current - trough) / trough if trough > 0 else 0

            if REQUIRE_STABILIZATION:
                r5 = series.tail(5)
                if len(r5) >= 5 and (float(r5.iloc[-1]) / float(r5.iloc[0]) - 1) < -0.03:
                    continue

            row = quality_df[quality_df["ticker"] == ticker].iloc[0]
            q = 0
            if row["roe"] and row["roe"] > 0:
                q += min(row["roe"] / 0.30, 1.0) * 40
            if row["grossMargins"] and row["grossMargins"] > 0:
                q += min(row["grossMargins"] / 0.60, 1.0) * 30
            q += (max(0, (150 - row["debtToEquity"]) / 150) * 30
                  if row["debtToEquity"] is not None else 15)

            fallen.append({
                **row.to_dict(),
                "peak_price": peak, "current_price": current, "trough_price": trough,
                "drop_from_peak": drop_pct, "bounce_from_trough": bounce,
                "days_since_peak": days_since,
                "peak_date": peak_date.strftime("%Y-%m-%d") if hasattr(peak_date, "strftime") else str(peak_date),
                "quality_score": q,
                "opportunity_score": q * abs(drop_pct) * (1 + bounce),
            })
        except Exception:
            continue

    df = pd.DataFrame(fallen)
    if len(df) > 0:
        df = df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    print(f"    Fallen angels: {len(df)}")
    return df


# ════════════════════════════════════════════════════════
#  PHASE 2 : DEEP DATA FETCH
# ════════════════════════════════════════════════════════

def fetch_macro():
    """주요 거시경제 지표 수집 + S&P500 200일선 체크"""
    symbols = {
        "S&P 500":    "^GSPC",
        "NASDAQ":     "^IXIC",
        "VIX":        "^VIX",
        "10Y 국채금리": "^TNX",
        "달러인덱스":   "DX-Y.NYB",
        "금":          "GC=F",
        "WTI 원유":    "CL=F",
    }
    result = {}
    try:
        raw = yf.download(list(symbols.values()), period="5d", progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        for name, sym in symbols.items():
            try:
                if sym not in close.columns:
                    continue
                series = close[sym].dropna()
                if len(series) < 1:
                    continue
                cur  = float(series.iloc[-1])
                prev = float(series.iloc[-2]) if len(series) >= 2 else cur
                chg  = (cur - prev) / prev if prev else 0
                result[name] = {"symbol": sym, "price": cur, "change": chg}
            except Exception:
                continue
    except Exception:
        pass

    # S&P500 200일선 별도 체크 (1년 데이터 필요)
    try:
        sp_hist = yf.Ticker("^GSPC").history(period="1y")
        if len(sp_hist) >= 200 and "S&P 500" in result:
            ma200 = float(sp_hist["Close"].rolling(200).mean().iloc[-1])
            result["S&P 500"]["ma200"]       = ma200
            result["S&P 500"]["above_ma200"] = result["S&P 500"]["price"] > ma200
    except Exception:
        pass

    return result


def assess_macro_regime(macro):
    """거시경제 지표 → 리스크 조정값 산출"""
    regime = {
        "position_multiplier": 1.0,
        "score_penalty":       0,
        "growth_penalty":      0,
        "bearish_market":      False,
        "warnings":            [],
        "notes":               [],
    }
    if not macro:
        return regime

    # VIX 공포지수
    vix_d = macro.get("VIX")
    if vix_d:
        v = vix_d["price"]
        if v >= 35:
            regime["position_multiplier"] *= 0.50
            regime["score_penalty"]       += 10
            regime["warnings"].append(f"VIX {v:.0f} 극도의 공포 — 포지션 50% 축소 / 전종목 -10점")
        elif v >= 25:
            regime["position_multiplier"] *= 0.75
            regime["score_penalty"]       += 5
            regime["warnings"].append(f"VIX {v:.0f} 공포 구간 — 포지션 25% 축소 / 전종목 -5점")
        elif v < 15:
            regime["notes"].append(f"VIX {v:.0f} 탐욕 구간 — 시장 안정, 정상 포지션 유지")

    # 10Y 국채금리
    tnx_d = macro.get("10Y 국채금리")
    if tnx_d:
        rate, chg = tnx_d["price"], tnx_d["change"]
        if rate > 4.5 and chg > 0.005:
            regime["growth_penalty"] += 15
            regime["warnings"].append(f"10Y 금리 {rate:.2f}% 급등 중 — 고PER 성장주 -15점")
        elif rate > 4.5:
            regime["growth_penalty"] += 8
            regime["warnings"].append(f"10Y 금리 {rate:.2f}% 고금리 — 고PER 성장주 -8점")
        elif rate < 3.5:
            regime["notes"].append(f"10Y 금리 {rate:.2f}% 저금리 — 성장주 우호 환경")

    # S&P500 200일선
    sp_d = macro.get("S&P 500")
    if sp_d:
        if sp_d.get("above_ma200") is False:
            regime["bearish_market"]      = True
            regime["position_multiplier"] *= 0.60
            regime["score_penalty"]       += 15
            regime["warnings"].append(
                f"S&P500({sp_d['price']:,.0f}) 200일선({sp_d.get('ma200',0):,.0f}) 하회"
                " — 약세장, 포지션 40% 추가 축소 / 전종목 -15점"
            )
        elif sp_d.get("above_ma200") is True:
            regime["notes"].append(
                f"S&P500({sp_d['price']:,.0f}) 200일선({sp_d.get('ma200',0):,.0f}) 상회 — 강세장 기조"
            )
        if sp_d["change"] < -0.02:
            regime["score_penalty"] += 5
            regime["warnings"].append(f"S&P500 당일 {sp_d['change']:.1%} 급락 — 전종목 추가 -5점")

    # 달러인덱스
    dxy_d = macro.get("달러인덱스")
    if dxy_d and dxy_d["change"] > 0.005:
        regime["notes"].append(f"달러인덱스 강세({dxy_d['change']:+.1%}) — 수출주·원자재주 주의")

    return regime


def _news_url(content):
    return ((content.get("canonicalUrl") or {}).get("url") or
            (content.get("clickThroughUrl") or {}).get("url") or "")


def fetch_market_news():
    """SPY 뉴스로 시장 전반 헤드라인 수집"""
    try:
        news = yf.Ticker("SPY").news or []
        result = []
        for n in news[:6]:
            c = n.get("content", {})
            title = c.get("title", "")
            if not title:
                continue
            result.append({"title": title,
                            "date":   c.get("pubDate", "")[:10],
                            "source": c.get("provider", {}).get("displayName", ""),
                            "url":    _news_url(c)})
        return result
    except Exception:
        return []


def fetch_news(ticker):
    try:
        news = yf.Ticker(ticker).news or []
        result = []
        for n in news[:6]:
            c = n.get("content", {})
            result.append({"title":  c.get("title", ""),
                            "date":   c.get("pubDate", ""),
                            "source": c.get("provider", {}).get("displayName", ""),
                            "url":    _news_url(c)})
        return result
    except Exception:
        return []


def fetch_insider(ticker):
    try:
        ins = yf.Ticker(ticker).insider_transactions
        if ins is None or len(ins) == 0:
            return {"has_data": False, "net_signal": "NO_DATA", "buys": 0, "sells": 0, "details": []}
        buys = sells = 0
        details = []
        for _, row in ins.head(15).iterrows():
            txt   = str(row.get("Text", "")).lower()
            name  = str(row.get("Insider", ""))
            shares = row.get("Shares", 0)
            if any(k in txt for k in ("purchase", "buy", "acquisition")):
                buys += 1; details.append(f"BUY: {name} ({shares:,.0f})")
            elif any(k in txt for k in ("sale", "sell")):
                sells += 1; details.append(f"SELL: {name} ({shares:,.0f})")
        return {"has_data": True,
                "net_signal": "BUYING" if buys > sells else "SELLING" if sells > buys else "NEUTRAL",
                "buys": buys, "sells": sells, "details": details[:5]}
    except Exception:
        return {"has_data": False, "net_signal": "NO_DATA", "buys": 0, "sells": 0, "details": []}


def fetch_analyst(ticker):
    try:
        info = yf.Ticker(ticker).info
        cur  = info.get("currentPrice") or info.get("regularMarketPrice")
        tm   = info.get("targetMeanPrice")
        return {
            "recommendation": info.get("recommendationKey", "none"),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
            "target_mean": tm, "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"), "current_price": cur,
            "upside_to_mean": (tm - cur) / cur if tm and cur and cur > 0 else None,
        }
    except Exception:
        return {"recommendation": "unknown", "num_analysts": 0}


def fetch_technicals(ticker):
    try:
        hist  = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 50: return {}
        close = hist["Close"]
        cur   = float(close.iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200= float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])))
        sma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        vol   = hist["Volume"]
        return {
            "current": cur, "sma50": sma50, "sma200": sma200,
            "high_52w": float(close.max()), "low_52w": float(close.min()),
            "rsi": rsi, "bb_lower": float(sma20 - 2 * std20),
            "vol_ratio": float(vol.tail(10).mean()) / max(float(vol.tail(50).mean()), 1),
            "above_sma50": cur > sma50,
            "above_sma200": cur > sma200 if sma200 else None,
            "pct_from_52w_high": (cur - float(close.max())) / float(close.max()),
        }
    except Exception:
        return {}


# ════════════════════════════════════════════════════════
#  PHASE 3 : SCORING
# ════════════════════════════════════════════════════════

def assess_drop_nature(row, insider, analyst, technicals):
    T, S = [], []
    rg, eg, fpe, tpe = row.get("revenueGrowth"), row.get("earningsGrowth"), row.get("forwardPE"), row.get("trailingPE")
    if rg is not None:
        (S if rg < 0 else (S if rg < 0.05 else T)).append(f"Revenue {'declining' if rg<0 else 'stalling'} ({rg:.1%})")
    if eg is not None:
        (S if eg < -0.20 else (T if eg < 0 else [])).append(f"Earnings {'contracting' if eg<-0.2 else 'dip'} ({eg:.1%})")
    if fpe and tpe and fpe < tpe: T.append("Fwd PE < Trailing PE (improving earnings)")
    if fpe and fpe > 40:          S.append(f"Still expensive (Fwd PE {fpe:.0f}x)")
    sig = insider.get("net_signal")
    if sig == "BUYING":           T.append(f"Insider BUYING ({insider['buys']} buys)")
    elif sig == "SELLING" and insider.get("sells", 0) >= 3:
                                  S.append(f"Heavy insider selling ({insider['sells']} sells)")
    rec = analyst.get("recommendation", "")
    if rec in ("strong_buy", "buy"):   T.append(f"Analysts bullish ({rec})")
    elif rec in ("sell", "strong_sell"): S.append(f"Analyst: {rec}")
    up = analyst.get("upside_to_mean")
    if up and up > 0.40: T.append(f"Large upside to target (+{up:.0%})")
    if technicals:
        rsi = technicals.get("rsi", 50)
        if rsi < 30:                T.append(f"RSI {rsi:.0f} (oversold)")
        if technicals.get("above_sma50"): T.append("Above 50-day MA")
    b = row.get("bounce_from_trough", 0)
    if b > 0.10:   T.append(f"Strong bounce +{b:.1%}")
    elif b <= 0:   S.append("No bounce — still falling")
    if abs(row.get("drop_from_peak", 0)) < 0.25: T.append("Moderate drop (<25%) — likely rotation")

    tc, sc = len(T), len(S)
    nature = "TEMPORARY" if sc==0 else "STRUCTURAL" if sc>=3 else "MOSTLY_TEMPORARY" if tc>=sc*2 else "MIXED"
    return {"nature": nature, "temporary_factors": T, "structural_factors": S}


def score_candidate(row, insider, analyst, technicals, assessment, macro_regime=None):
    score, reasons, warnings = 0, [], []
    q = row.get("quality_score", 0)
    score += min(q / 100 * 25, 25)
    if q >= 80: reasons.append(f"Quality {q:.0f}/100 (top tier)")

    fpe = row.get("forwardPE")
    if fpe and fpe > 0:
        pts = 20 if fpe<12 else 14 if fpe<20 else 8 if fpe<30 else 3
        score += pts
        if fpe < 12:  reasons.append(f"Fwd PE {fpe:.1f} (cheap)")
        elif fpe > 30: warnings.append(f"Fwd PE {fpe:.1f} (expensive)")

    b = row.get("bounce_from_trough", 0)
    if b>0.10:   score+=15; reasons.append(f"Strong bounce +{b:.1%}")
    elif b>0.05: score+=12; reasons.append(f"Bounce +{b:.1%}")
    elif b>0.02: score+=8
    elif b>0:    score+=4
    else:        warnings.append("No bounce — falling knife risk")

    if insider.get("has_data"):
        sig = insider["net_signal"]
        if sig == "BUYING":   score+=15; reasons.append(f"Insider BUYING ({insider['buys']} buys)")
        elif sig == "NEUTRAL": score+=8
        else:                 score+=2;  warnings.append(f"Insider selling ({insider['sells']} sells)")
    else:
        score += 5

    rec = analyst.get("recommendation", "")
    up  = analyst.get("upside_to_mean")
    if rec in ("strong_buy","buy"):    score+=10; reasons.append(f"Analyst: {rec} ({analyst.get('num_analysts',0)})")
    elif rec == "hold":                score+=5
    elif rec in ("sell","strong_sell"): warnings.append(f"Analyst: {rec}")
    if up and up > 0.30:               score+=3;  reasons.append(f"Upside to target: +{up:.0%}")

    if technicals:
        rsi = technicals.get("rsi", 50)
        if rsi<30:   score+=10; reasons.append(f"RSI {rsi:.0f} (oversold)")
        elif rsi<40: score+=7
        elif rsi<50: score+=4
        else:        score+=1
        if technicals.get("above_sma50"):       score+=3; reasons.append("Above 50-day MA")
        if technicals.get("vol_ratio",1) > 1.5: score+=2

    nat = assessment.get("nature","MIXED")
    if nat == "STRUCTURAL":
        score = int(score*0.50); warnings.append("Structural decline — score ×0.5")
    elif nat == "MIXED":
        score = int(score*0.75); warnings.append("Mixed signals — score ×0.75")

    # 매크로 페널티 적용
    if macro_regime:
        penalty = macro_regime["score_penalty"]
        fpe_val = row.get("forwardPE")
        if fpe_val and fpe_val > 30:
            penalty += macro_regime["growth_penalty"]
            if macro_regime["growth_penalty"] > 0:
                warnings.append(f"고PER({fpe_val:.0f}x) × 금리 페널티 -{macro_regime['growth_penalty']}점")
        if penalty > 0:
            score = max(0, score - penalty)
            warnings.append(f"매크로 환경 페널티 -{penalty}점 적용")

    grade = "STRONG BUY" if score>=60 else "BUY" if score>=45 else "WATCH" if score>=30 else "PASS"
    return {"score": min(score,100), "grade": grade, "reasons": reasons, "warnings": warnings}


# ════════════════════════════════════════════════════════
#  PHASE 4 : TRADE PLAN
# ════════════════════════════════════════════════════════

def calculate_trade_plan(grade, cur, peak, trough, tech, analyst, macro_regime=None):
    if grade == "PASS" or cur <= 0:
        return {"action":"DO NOT BUY","shares":0,"amount":0,"pct_portfolio":0,
                "stop_loss":0,"stop_pct":0,"targets":[],"trailing_stop":0,
                "risk_reward":0,"risk_amount":0}
    rm = {"STRONG BUY":1.5,"BUY":1.0,"WATCH":0.5}[grade]
    tr = {"STRONG BUY":0.08,"BUY":0.10,"WATCH":0.12}[grade]
    entry = {"STRONG BUY":"Full position now","BUY":"Half position, add on dip","WATCH":"Wait for catalyst"}[grade]

    stop = max([s for s in [trough*0.97, cur*0.92,
                             (tech.get("bb_lower",0)*0.98 if tech else 0),
                             ((analyst.get("target_low") or 0)*0.90)] if s>0],
               default=cur*0.92)
    stop = min(stop, cur*0.95)

    rng = peak - trough
    t1, t2, t3 = trough+rng*0.50, trough+rng*0.75, trough+rng*0.90
    at = analyst.get("target_mean")
    if at and at > t2: t3 = max(t3, at*0.95)
    targets = [
        {"price":round(t1,2),"pct":(t1-cur)/cur,"action":"Sell 30%",         "label":"50% 회복"},
        {"price":round(t2,2),"pct":(t2-cur)/cur,"action":"Sell 30%",         "label":"75% 회복"},
        {"price":round(t3,2),"pct":(t3-cur)/cur,"action":"Sell remaining 40%","label":"90% 회복"},
    ]
    rps   = max(cur - stop, cur*0.05)
    pm    = macro_regime["position_multiplier"] if macro_regime else 1.0
    shares = min(int(TOTAL_CAPITAL*RISK_PER_TRADE*rm*pm / rps),
                 int(TOTAL_CAPITAL*MAX_POSITION_PCT*pm / cur))
    amount = shares * cur
    return {
        "action":entry, "shares":shares, "amount":round(amount,2),
        "pct_portfolio": amount/TOTAL_CAPITAL,
        "stop_loss":round(stop,2), "stop_pct":(stop-cur)/cur,
        "targets":targets, "trailing_stop":tr,
        "risk_reward":round((t2-cur)/(cur-stop),1) if cur>stop else 0,
        "risk_amount":round(shares*rps,2),
    }


# ════════════════════════════════════════════════════════
#  PHASE 5 : DAY-OVER-DAY COMPARISON
# ════════════════════════════════════════════════════════

def load_previous():
    today = datetime.now().strftime("%Y%m%d")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "analysis_*.csv")), reverse=True)
    files = [f for f in files if today not in f]
    if not files: return None, None
    try:
        df = pd.read_csv(files[0])
        date_str = files[0].split("analysis_")[-1].replace(".csv","")
        return df, date_str
    except Exception:
        return None, None


def build_comparison(today_results, prev_df, prev_date):
    if prev_df is None or len(prev_df) == 0:
        return None
    tm = {r["ticker"]: r for r in today_results}
    pm = {row["ticker"]: row for _, row in prev_df.iterrows()}
    today_t, prev_t = set(tm), set(pm)
    go = {"STRONG BUY":4,"BUY":3,"WATCH":2,"PASS":1}
    rows = []
    for tk in sorted(today_t | prev_t):
        t, p = tm.get(tk), pm.get(tk)
        if t is not None and p is not None:
            tg, pg = t["verdict"]["grade"], str(p.get("grade",""))
            gd = go.get(tg,0) - go.get(pg,0)
            rows.append({"ticker":tk,"name":t["name"],"sector":t["sector"],
                "today_grade":tg,"prev_grade":pg,
                "grade_change":"UP" if gd>0 else "DOWN" if gd<0 else "SAME",
                "today_score":t["verdict"]["score"],"prev_score":float(p.get("score",0)),
                "score_delta":t["verdict"]["score"]-float(p.get("score",0)),
                "today_price":t["current_price"],"prev_price":float(p.get("price",t["current_price"])),
                "price_chg":(t["current_price"]-float(p.get("price",t["current_price"])))/float(p.get("price",t["current_price"])) if float(p.get("price",0))>0 else 0,
                "today_nature":t["assessment"]["nature"],
                "today_drop":t["drop"],"today_bounce":t["bounce"],
                "is_new":False,"is_exit":False})
        elif t is not None:
            rows.append({"ticker":tk,"name":t["name"],"sector":t["sector"],
                "today_grade":t["verdict"]["grade"],"prev_grade":"-","grade_change":"NEW",
                "today_score":t["verdict"]["score"],"prev_score":0,"score_delta":t["verdict"]["score"],
                "today_price":t["current_price"],"prev_price":0,"price_chg":0,
                "today_nature":t["assessment"]["nature"],
                "today_drop":t["drop"],"today_bounce":t["bounce"],
                "is_new":True,"is_exit":False})
        else:
            rows.append({"ticker":tk,"name":str(p.get("name","")),"sector":str(p.get("sector","")),
                "today_grade":"-","prev_grade":str(p.get("grade","")),"grade_change":"EXIT",
                "today_score":0,"prev_score":float(p.get("score",0)),"score_delta":-float(p.get("score",0)),
                "today_price":0,"prev_price":float(p.get("price",0)),"price_chg":0,
                "today_nature":"-","today_drop":0,"today_bounce":0,
                "is_new":False,"is_exit":True})

    order = {"NEW":0,"UP":1,"SAME":2,"DOWN":3,"EXIT":4}
    rows.sort(key=lambda x: (order.get(x["grade_change"],9), -x["today_score"]))
    return {"rows":rows,"prev_date":prev_date,
            "new_count":sum(1 for r in rows if r["grade_change"]=="NEW"),
            "exit_count":sum(1 for r in rows if r["grade_change"]=="EXIT"),
            "upgraded":sum(1 for r in rows if r["grade_change"]=="UP"),
            "downgraded":sum(1 for r in rows if r["grade_change"]=="DOWN")}


# ════════════════════════════════════════════════════════
#  PHASE 6 : HTML REPORT
# ════════════════════════════════════════════════════════

GC  = {"STRONG BUY":"#0F6E56","BUY":"#185FA5","WATCH":"#854F0B","PASS":"#A32D2D"}
GB  = {"STRONG BUY":"#E1F5EE","BUY":"#E6F1FB","WATCH":"#FAEEDA","PASS":"#FCEBEB"}
NS  = {"TEMPORARY":("#E1F5EE","#0F6E56"),"MOSTLY_TEMPORARY":("#E6F1FB","#185FA5"),
       "MIXED":("#FAEEDA","#854F0B"),"STRUCTURAL":("#FCEBEB","#A32D2D")}

def bdg(text, bg, fg):
    return f'<span class="badge" style="background:{bg};color:{fg}">{text}</span>'
def gbdg(g):
    return bdg(g, GB.get(g,"#eee"), GC.get(g,"#888"))
def nbdg(n):
    bg,fg = NS.get(n,("#eee","#888")); return bdg(n,bg,fg)

CHG_BADGE = {
    "NEW":  bdg("🆕 NEW",  "#E1F5EE", "#0F6E56"),
    "EXIT": bdg("🚪 EXIT", "#FCEBEB", "#A32D2D"),
    "UP":   bdg("⬆ UP",   "#E1F5EE", "#0F6E56"),
    "DOWN": bdg("⬇ DOWN", "#FCEBEB", "#A32D2D"),
    "SAME": bdg("— SAME", "#f1f3f5", "#888"),
}


def comparison_section(comp):
    if not comp:
        return '<p style="color:#888;padding:16px;background:#f8f9fa;border-radius:8px">전일 데이터 없음 (첫 실행)</p>'
    pd_fmt = f'{comp["prev_date"][:4]}-{comp["prev_date"][4:6]}-{comp["prev_date"][6:]}'
    rows_h = ""
    for c in comp["rows"]:
        chg = c["grade_change"]
        sd  = c["score_delta"]
        sc  = "#0F6E56" if sd>0 else "#A32D2D" if sd<0 else "#888"
        sign = "+" if sd>=0 else ""
        pc  = c["price_chg"]
        pcol = "#0F6E56" if pc>0 else "#A32D2D" if pc<0 else "#888"
        psign = "+" if pc>=0 else ""
        price_cell = (f'<td>${c["today_price"]:.2f} <small style="color:{pcol}">({psign}{pc:.1%})</small></td>'
                      if c["today_price"]>0 else "<td>—</td>")
        rows_h += f"""<tr class="chg-{chg.lower()}">
            <td><b>{c["ticker"]}</b></td>
            <td class="muted">{c["name"][:20]}</td>
            <td>{CHG_BADGE.get(chg,"")}</td>
            <td>{gbdg(c["prev_grade"]) if c["prev_grade"] not in ("-","") else "<span class='muted'>—</span>"}</td>
            <td>{gbdg(c["today_grade"]) if c["today_grade"] not in ("-","") else "<span class='muted'>—</span>"}</td>
            <td><b>{c["today_score"]:.0f}</b></td>
            <td style="color:{sc}">{sign}{sd:.1f}</td>
            {price_cell}
            <td>{nbdg(c["today_nature"]) if c["today_nature"]!="-" else ""}</td>
            <td style="color:#A32D2D">{c["today_drop"]:.1%}</td>
            <td style="color:#0F6E56">{c["today_bounce"]:+.1%}</td>
        </tr>"""
    return f"""
    <div class="stat-grid four">
        <div class="stat" style="border-left:3px solid #0F6E56"><div class="sl">🆕 신규 진입</div><div class="sv green">{comp["new_count"]}</div></div>
        <div class="stat" style="border-left:3px solid #A32D2D"><div class="sl">🚪 이탈</div><div class="sv red">{comp["exit_count"]}</div></div>
        <div class="stat" style="border-left:3px solid #0F6E56"><div class="sl">⬆ 등급 상향</div><div class="sv green">{comp["upgraded"]}</div></div>
        <div class="stat" style="border-left:3px solid #A32D2D"><div class="sl">⬇ 등급 하향</div><div class="sv red">{comp["downgraded"]}</div></div>
    </div>
    <div class="table-wrap">
    <table>
        <thead><tr><th>Ticker</th><th>Name</th><th>변화</th><th>전일 등급</th><th>오늘 등급</th>
        <th>Score</th><th>Score Δ</th><th>현재가 (전일比)</th><th>하락 성격</th><th>낙폭</th><th>반등</th></tr></thead>
        <tbody>{rows_h}</tbody>
    </table>
    </div>
    <p class="muted" style="font-size:12px;margin:6px 0">전일 기준: {pd_fmt}</p>"""


def card(r):
    v, tp, a, ins, an, tech = r["verdict"], r["trade_plan"], r["assessment"], r["insider"], r["analyst"], r["technicals"]
    reasons_h  = "".join(f'<div class="signal-pos">+ {x}</div>' for x in v["reasons"])
    warnings_h = "".join(f'<div class="signal-neg">! {x}</div>' for x in v["warnings"])
    temp_h     = "".join(f'<div class="signal-pos" style="font-size:13px">+ {x}</div>' for x in a["temporary_factors"])
    struct_h   = "".join(f'<div class="signal-neg" style="font-size:13px">- {x}</div>' for x in a["structural_factors"])
    news_h     = "".join(
        f'<div class="news-item">'
        f'<a href="{n.get("url","#") or "#"}" target="_blank" rel="noopener noreferrer" '
        f'style="color:inherit;text-decoration:none;font-weight:500">'
        f'{n["title"][:80]}{"…" if len(n["title"])>80 else ""}</a>'
        f'<small class="muted"> · {n["source"]} · {n["date"][:10]}</small>'
        f'</div>'
        for n in r["news"][:5]
    )
    sig_col    = "#0F6E56" if ins.get("net_signal")=="BUYING" else "#A32D2D" if ins.get("net_signal")=="SELLING" else "#888"
    ins_h      = (f'<span style="color:{sig_col};font-weight:500">{ins["net_signal"]}</span> (매수 {ins["buys"]} / 매도 {ins["sells"]})'
                  + "".join(f'<div class="muted" style="font-size:12px">{d}</div>' for d in ins.get("details",[])[:3])
                  if ins.get("has_data") else '<span class="muted">데이터 없음</span>')

    if tp["shares"] > 0:
        tgt_rows = "".join(f'<tr><td>{t["label"]}</td><td><code>${t["price"]:.2f}</code></td>'
                           f'<td class="green">+{t["pct"]:.1%}</td><td class="muted">{t["action"]}</td></tr>'
                           for t in tp["targets"])
        trade_h = f"""
        <div class="mini-grid four">
            <div class="mini-stat"><div class="sl">수량</div><div class="sv">{tp["shares"]}</div></div>
            <div class="mini-stat"><div class="sl">금액</div><div class="sv">${tp["amount"]:,.0f}</div></div>
            <div class="mini-stat"><div class="sl">비중</div><div class="sv">{tp["pct_portfolio"]:.1%}</div></div>
            <div class="mini-stat"><div class="sl">R/R</div><div class="sv">1:{tp["risk_reward"]}</div></div>
        </div>
        <table class="trade-table">
            <tr class="stop-row"><td>손절선</td><td><code>${tp["stop_loss"]:.2f}</code></td>
                <td class="red">{tp["stop_pct"]:.1%}</td><td class="muted">전량 매도</td></tr>
            {tgt_rows}
        </table>
        <p class="muted" style="font-size:12px">목표1 달성 후 잔여 포지션에 {tp["trailing_stop"]:.0%} 트레일링 스톱 발동</p>"""
    else:
        trade_h = f'<div class="no-trade">{tp["action"]}</div>'

    rsi = tech.get("rsi",0) if tech else 0
    rsi_c = "#0F6E56" if rsi<30 else "#A32D2D" if rsi>70 else "#555"
    tech_h = (f'<div class="tech-grid">'
              f'<span>RSI: <b style="color:{rsi_c}">{rsi:.0f}</b></span>'
              f'<span>50일선: <b>{"위" if tech.get("above_sma50") else "아래"}</b></span>'
              f'<span>52주고점比: {tech.get("pct_from_52w_high",0):.0%}</span>'
              f'</div>') if tech else '<span class="muted">N/A</span>'

    an_tl = f'${an["target_low"]:.2f}' if isinstance(an.get("target_low"),(int,float)) else "?"
    an_tm = f'${an["target_mean"]:.2f}' if isinstance(an.get("target_mean"),(int,float)) else "?"
    an_th = f'${an["target_high"]:.2f}' if isinstance(an.get("target_high"),(int,float)) else "?"
    an_up = f'<br>상승여력: <b class="green">+{an["upside_to_mean"]:.0%}</b>' if an.get("upside_to_mean") else ""
    fpe   = r.get("fwd_pe","N/A")

    return f"""
    <div class="card" style="border-left-color:{GC.get(v['grade'],'#888')}">
        <div class="card-header">
            <div>
                <span class="ticker">{r["ticker"]}</span>
                <span class="muted"> {r["name"]}</span>
                <span class="sector-tag">{r["sector"]}</span>
            </div>
            <div class="card-badges">
                {nbdg(a['nature'])}
                {gbdg(v['grade'])}
                <span class="score-num">{v['score']:.0f}</span>
            </div>
        </div>
        <div class="kpi-grid">
            <div class="kpi"><div class="kl">현재가</div><div class="kv">${r["current_price"]:.2f}</div></div>
            <div class="kpi"><div class="kl">낙폭</div><div class="kv red">{r["drop"]:.1%}</div></div>
            <div class="kpi"><div class="kl">반등</div><div class="kv green">{r["bounce"]:+.1%}</div></div>
            <div class="kpi"><div class="kl">ROE</div><div class="kv">{r["roe"]:.1%}</div></div>
            <div class="kpi"><div class="kl">Fwd PE</div><div class="kv">{fpe}</div></div>
        </div>
        <details><summary>하락 원인 분석</summary><div class="detail-body">{temp_h}{struct_h}</div></details>
        <details><summary>매수/매도 시그널</summary><div class="detail-body">{reasons_h}{warnings_h}</div></details>
        <details><summary>내부자 거래</summary><div class="detail-body">{ins_h}</div></details>
        <details><summary>애널리스트 컨센서스</summary>
            <div class="detail-body">
                추천: <b>{an.get("recommendation","N/A")}</b> ({an.get("num_analysts",0)}명)<br>
                목표가: {an_tl} ~ {an_tm} ~ {an_th}{an_up}
            </div>
        </details>
        <details><summary>기술적 분석</summary><div class="detail-body">{tech_h}</div></details>
        <details><summary>최근 뉴스</summary><div class="detail-body news-list">{news_h or "<span class='muted'>없음</span>"}</div></details>
        <div class="trade-section">
            <div class="trade-title">트레이드 플랜</div>
            {trade_h}
        </div>
    </div>"""


def vix_label(v):
    if v < 15:   return ("극도의 탐욕", "#0F6E56")
    if v < 20:   return ("탐욕",       "#3a8c3f")
    if v < 25:   return ("중립",       "#888")
    if v < 30:   return ("공포",       "#c47900")
    return              ("극도의 공포", "#A32D2D")


def macro_section(macro, market_news):
    if not macro:
        return ""

    def tile(name, data):
        p    = data["price"]
        chg  = data["change"]
        sign = "+" if chg >= 0 else ""
        col  = "#0F6E56" if chg >= 0 else "#A32D2D"
        arrow = "▲" if chg >= 0 else "▼"

        # VIX 전용: 공포 레이블 추가
        extra = ""
        if name == "VIX":
            lbl, lc = vix_label(p)
            extra = f'<div class="macro-fear" style="color:{lc}">{lbl}</div>'

        # 단위 포맷
        if name in ("10Y 국채금리",):
            price_fmt = f"{p:.2f}%"
        elif name == "달러인덱스":
            price_fmt = f"{p:.1f}"
        else:
            price_fmt = f"${p:,.2f}" if p >= 10 else f"{p:.2f}"

        return f"""<div class="macro-tile">
            <div class="macro-name">{name}</div>
            <div class="macro-price">{price_fmt}</div>
            <div class="macro-chg" style="color:{col}">{arrow} {sign}{chg:.2%}</div>
            {extra}
        </div>"""

    tiles_h = "".join(tile(n, d) for n, d in macro.items())

    news_h = ""
    if market_news:
        items = "".join(
            f'<div class="mkt-news-item">'
            f'<a class="mkt-news-title" href="{n.get("url","#") or "#"}" target="_blank" rel="noopener noreferrer" '
            f'style="color:inherit;text-decoration:none;font-weight:500">'
            f'{n["title"][:90]}{"…" if len(n["title"])>90 else ""}</a>'
            f'<span class="muted" style="font-size:11px;white-space:nowrap"> · {n["source"]} {n["date"]}</span>'
            f'</div>'
            for n in market_news[:5]
        )
        news_h = f'<div class="mkt-news-wrap"><div class="mkt-news-header">📰 시장 주요 뉴스</div>{items}</div>'

    return f"""
<div class="macro-section">
  <div class="macro-header">📊 오늘의 시장 현황</div>
  <div class="macro-grid">{tiles_h}</div>
  {news_h}
</div>"""


def regime_banner(macro_regime):
    if not macro_regime:
        return ""
    pm   = macro_regime["position_multiplier"]
    warn = macro_regime["warnings"]
    note = macro_regime["notes"]
    if not warn and not note:
        return ""

    pm_pct = int(pm * 100)
    pm_col = "#0F6E56" if pm >= 1.0 else "#c47900" if pm >= 0.75 else "#A32D2D"

    warn_h = "".join(f'<div class="regime-warn">⚠ {w}</div>' for w in warn)
    note_h = "".join(f'<div class="regime-note">✓ {n}</div>' for n in note)

    return f"""
<div class="regime-box" style="border-color:{pm_col}">
  <div class="regime-header" style="color:{pm_col}">
    매크로 리스크 분석 &nbsp;·&nbsp; 포지션 배수:
    <strong>{pm_pct}%</strong>
    {"(정상)" if pm>=1.0 else "(축소 적용 중)"}
  </div>
  {warn_h}{note_h}
</div>"""


def generate_html(results, comparison, date_str, macro=None, market_news=None, macro_regime=None):
    actionable  = [r for r in results if r["verdict"]["grade"] in ("STRONG BUY","BUY")]
    total_alloc = sum(r["trade_plan"]["amount"] for r in actionable)
    total_risk  = sum(r["trade_plan"].get("risk_amount",0) for r in actionable)
    date_fmt    = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    run_time    = datetime.now().strftime("%Y-%m-%d %H:%M")

    port_rows = "".join(f"""<tr>
        <td><b>{r["ticker"]}</b></td><td>{gbdg(r["verdict"]["grade"])}</td>
        <td>{r["trade_plan"]["shares"]}</td><td>${r["trade_plan"]["amount"]:,.0f}</td>
        <td>{r["trade_plan"]["pct_portfolio"]:.1%}</td>
        <td class="red">${r["trade_plan"]["stop_loss"]:.2f} ({r["trade_plan"]["stop_pct"]:.1%})</td>
        <td class="green">${r["trade_plan"]["targets"][0]["price"]:.2f} (+{r["trade_plan"]["targets"][0]["pct"]:.1%})</td>
        <td>1:{r["trade_plan"]["risk_reward"]}</td>
    </tr>""" for r in actionable if r["trade_plan"]["shares"]>0)

    cards_h = "".join(card(r) for r in results)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fallen Angel Report · {date_fmt}</title>
<style>
:root{{--green:#0F6E56;--blue:#185FA5;--amber:#854F0B;--red:#A32D2D;--border:#e5e7eb;--bg:#f8f9fa;--radius:10px}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI",sans-serif;max-width:1000px;margin:0 auto;padding:24px 16px;color:#1a1a1a;line-height:1.6;background:#fff}}
h1{{font-size:22px;font-weight:700}}
h2{{font-size:16px;font-weight:600;margin:28px 0 12px;padding:0 0 8px;border-bottom:2px solid var(--border)}}
.muted{{color:#888}} .green{{color:var(--green)}} .red{{color:var(--red)}}
.badge{{padding:2px 9px;border-radius:6px;font-size:12px;font-weight:500;white-space:nowrap}}
/* Summary stats */
.stat-grid{{display:grid;gap:10px;margin:0 0 16px}}
.stat-grid.four{{grid-template-columns:repeat(4,1fr)}}
.stat-grid.two{{grid-template-columns:repeat(2,1fr)}}
.stat{{background:var(--bg);padding:12px 14px;border-radius:var(--radius)}}
.sl{{font-size:11px;text-transform:uppercase;color:#888;letter-spacing:.4px}}
.sv{{font-size:24px;font-weight:600;margin:2px 0}}
.sv.green{{color:var(--green)}} .sv.red{{color:var(--red)}}
/* Table */
.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:7px 10px;border-bottom:2px solid var(--border);color:#888;font-weight:500;font-size:12px;white-space:nowrap}}
td{{padding:6px 10px;border-bottom:1px solid #f0f0f0;white-space:nowrap}}
tr.chg-new{{background:#fafffe}} tr.chg-exit{{background:#fffafa}}
/* Cards */
.card{{border:1px solid var(--border);border-left:4px solid #888;border-radius:var(--radius);padding:18px;margin:0 0 14px}}
.card-header{{display:flex;justify-content:space-between;align-items:center;margin:0 0 12px;flex-wrap:wrap;gap:8px}}
.ticker{{font-size:20px;font-weight:700;font-family:monospace}}
.sector-tag{{font-size:11px;color:#aaa;margin-left:6px}}
.card-badges{{display:flex;align-items:center;gap:6px}}
.score-num{{font-size:18px;font-weight:600}}
.kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:0 0 12px}}
@media(max-width:600px){{.kpi-grid{{grid-template-columns:repeat(3,1fr)}}}}
.kpi{{background:var(--bg);padding:6px 10px;border-radius:6px}}
.kl{{font-size:11px;color:#888}} .kv{{font-size:15px;font-weight:500}}
/* Details */
details{{border-top:1px solid #f0f0f0;margin:0}} details:last-of-type{{border-bottom:1px solid #f0f0f0}}
details summary{{cursor:pointer;font-size:14px;font-weight:500;padding:8px 0;list-style:none;user-select:none}}
details summary::-webkit-details-marker{{display:none}}
details summary::before{{content:"▶ ";font-size:10px;color:#aaa}}
details[open] summary::before{{content:"▼ "}}
.detail-body{{padding:8px 0 10px;font-size:13px}}
.signal-pos{{color:var(--green);padding:2px 0}} .signal-neg{{color:var(--red);padding:2px 0}}
.news-item{{padding:4px 0;border-bottom:.5px solid #f0f0f0;font-size:13px}}
.tech-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
/* Trade plan */
.trade-section{{margin:12px 0 0;padding:12px 0 0;border-top:1px solid var(--border)}}
.trade-title{{font-weight:600;font-size:14px;margin:0 0 8px}}
.mini-grid{{display:grid;gap:8px;margin:0 0 8px}} .mini-grid.four{{grid-template-columns:repeat(4,1fr)}}
.mini-stat{{background:var(--bg);padding:8px 12px;border-radius:8px}}
.trade-table{{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}}
.trade-table td{{padding:4px 8px}}
.stop-row td{{background:#FCEBEB}} .stop-row td:nth-child(3){{color:var(--red)}}
.no-trade{{padding:12px;background:var(--bg);border-radius:8px;text-align:center;color:#888}}
code{{font-family:monospace;background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:13px}}
/* Header bar */
.header-bar{{display:flex;justify-content:space-between;align-items:flex-end;margin:0 0 20px;flex-wrap:wrap;gap:8px}}
.disclaimer{{font-size:12px;color:#888;margin:28px 0 0;padding:14px;background:var(--bg);border-radius:var(--radius)}}
/* Regime banner */
.regime-box{{border-left:4px solid #c47900;background:#fffdf5;border-radius:var(--radius);padding:14px 16px;margin:0 0 20px}}
.regime-header{{font-size:13px;font-weight:700;margin:0 0 8px;text-transform:uppercase;letter-spacing:.4px}}
.regime-warn{{font-size:13px;color:#7a3800;padding:2px 0;line-height:1.5}}
.regime-note{{font-size:13px;color:#1a5c38;padding:2px 0;line-height:1.5}}
/* Macro section */
.macro-section{{background:#f8f9fa;border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;margin:0 0 24px}}
.macro-header{{font-size:13px;font-weight:600;color:#555;margin:0 0 12px;text-transform:uppercase;letter-spacing:.5px}}
.macro-grid{{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin:0 0 14px}}
.macro-tile{{background:#fff;border:1px solid var(--border);border-radius:8px;padding:10px 12px;text-align:center}}
.macro-name{{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.4px;margin:0 0 4px}}
.macro-price{{font-size:15px;font-weight:600;font-family:monospace}}
.macro-chg{{font-size:12px;font-weight:500;margin:2px 0}}
.macro-fear{{font-size:10px;font-weight:600;margin:3px 0}}
.mkt-news-wrap{{border-top:1px solid var(--border);padding-top:12px}}
.mkt-news-header{{font-size:12px;font-weight:600;color:#555;margin:0 0 8px;text-transform:uppercase;letter-spacing:.4px}}
.mkt-news-item{{display:flex;justify-content:space-between;align-items:baseline;gap:8px;padding:4px 0;border-bottom:.5px solid #f0f0f0;font-size:13px}}
.mkt-news-title{{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
@media(max-width:700px){{
  .macro-grid{{grid-template-columns:repeat(3,1fr)}}
  .mkt-news-item{{flex-direction:column;gap:2px}}
}}
@media(max-width:600px){{
  .stat-grid.four{{grid-template-columns:repeat(2,1fr)}}
  .mini-grid.four{{grid-template-columns:repeat(2,1fr)}}
  body{{padding:12px 10px}}
  .macro-grid{{grid-template-columns:repeat(2,1fr)}}
}}
</style>
</head>
<body>

<div class="header-bar">
  <div>
    <h1>📈 Fallen Angel Daily Report</h1>
    <p class="muted" style="font-size:13px;margin:3px 0">
      {date_fmt} · 업데이트: {run_time} KST · 자본금 ${TOTAL_CAPITAL:,.0f} · S&amp;P 500 유니버스
    </p>
  </div>
  <div style="font-size:12px;color:#aaa;text-align:right">
    매일 오전 7시 자동 업데이트<br>
    <span style="color:var(--green)">●</span> 데이터: yfinance
  </div>
</div>

{macro_section(macro, market_news)}

{regime_banner(macro_regime)}

<h2>전일 대비 비교</h2>
{comparison_section(comparison)}

{'<h2>포트폴리오 요약</h2><div class="table-wrap"><table><thead><tr><th>Ticker</th><th>등급</th><th>수량</th><th>금액</th><th>비중</th><th>손절</th><th>목표 1</th><th>R/R</th></tr></thead><tbody>' + port_rows + '</tbody></table></div><p class="muted" style="font-size:12px;margin:6px 0">잔여현금: ${:,.0f} ({:.0%}) · 총 리스크: ${:,.0f} ({:.1%})</p>'.format(TOTAL_CAPITAL-total_alloc,(TOTAL_CAPITAL-total_alloc)/TOTAL_CAPITAL,total_risk,total_risk/TOTAL_CAPITAL) if actionable else ""}

<h2>종목별 상세 분석</h2>
{cards_h}

<div class="disclaimer">
본 리포트는 교육·분석 목적으로만 제공되며 투자 권유가 아닙니다.
모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
과거 성과는 미래 수익을 보장하지 않습니다.
</div>

</body>
</html>"""


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def main():
    date_str = datetime.now().strftime("%Y%m%d")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')} KST] Fallen Angel Daily Pipeline")
    print(f"  DOCS_DIR={DOCS_DIR}  DATA_DIR={DATA_DIR}")

    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Phase 1
    tickers    = get_sp500_tickers()
    quality_df = screen_quality(tickers)
    fallen_df  = screen_drops(quality_df)
    if len(fallen_df) == 0:
        print("  No fallen angels found."); return

    # Phase 2: 매크로 지표 수집 (분석 전에 먼저)
    print("  [2.5/5] Fetching macro indicators …")
    macro        = fetch_macro()
    market_news  = fetch_market_news()
    macro_regime = assess_macro_regime(macro)
    if macro_regime["warnings"]:
        for w in macro_regime["warnings"]:
            print(f"    ⚠ {w}")
    if macro_regime["notes"]:
        for n in macro_regime["notes"]:
            print(f"    ✓ {n}")

    # Phase 3-4
    analyze_n = min(ANALYZE_TOP_N, len(fallen_df))
    print(f"  [3/5] Analyzing top {analyze_n} …")
    results = []
    for idx, row in fallen_df.head(analyze_n).iterrows():
        ticker = row["ticker"]
        print(f"    [{idx+1}/{analyze_n}] {ticker} …", end=" ", flush=True)
        news       = fetch_news(ticker)
        insider    = fetch_insider(ticker)
        analyst    = fetch_analyst(ticker)
        tech       = fetch_technicals(ticker)
        assessment = assess_drop_nature(row, insider, analyst, tech)
        verdict    = score_candidate(row, insider, analyst, tech, assessment, macro_regime)
        cur_price  = tech.get("current", row.get("current_price", 0))
        trade_plan = calculate_trade_plan(verdict["grade"], cur_price,
                                          row.get("peak_price",0), row.get("trough_price",0),
                                          tech, analyst, macro_regime)
        fpe = row.get("forwardPE")
        results.append({
            "ticker":ticker,"name":row.get("name",""),"sector":row.get("sector",""),
            "current_price":cur_price,"drop":row.get("drop_from_peak",0),
            "bounce":row.get("bounce_from_trough",0),"roe":row.get("roe",0),
            "fwd_pe":f"{fpe:.1f}" if fpe else "N/A",
            "quality_score":row.get("quality_score",0),
            "news":news,"insider":insider,"analyst":analyst,
            "technicals":tech,"assessment":assessment,"verdict":verdict,"trade_plan":trade_plan,
        })
        icon = {"STRONG BUY":"++","BUY":"+ ","WATCH":"~ ","PASS":"x "}.get(verdict["grade"],"  ")
        print(f"{icon} {verdict['grade']} ({verdict['score']:.0f})")

    # Phase 5: comparison
    print("  [4/5] Comparing with previous day …")
    prev_df, prev_date = load_previous()
    comparison = build_comparison(results, prev_df, prev_date)

    # Phase 6: save
    print("  [5/5] Generating report …")
    html = generate_html(results, comparison, date_str, macro, market_news, macro_regime)
    html_path = os.path.join(DOCS_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    csv_rows = []
    for r in results:
        tp = r["trade_plan"]
        csv_rows.append({
            "ticker":r["ticker"],"name":r["name"],"sector":r["sector"],
            "grade":r["verdict"]["grade"],"score":r["verdict"]["score"],
            "nature":r["assessment"]["nature"],
            "price":r["current_price"],"drop":r["drop"],"bounce":r["bounce"],
            "roe":r["roe"],"fwd_pe":r["fwd_pe"],"quality":r["quality_score"],
            "analyst_rec":r["analyst"].get("recommendation"),
            "analyst_upside":r["analyst"].get("upside_to_mean"),
            "insider_signal":r["insider"].get("net_signal"),
            "rsi":r["technicals"].get("rsi"),
            "shares":tp["shares"],"amount":tp["amount"],
            "stop_loss":tp["stop_loss"],"stop_pct":tp["stop_pct"],
            "target_1":tp["targets"][0]["price"] if tp["targets"] else 0,
            "target_2":tp["targets"][1]["price"] if len(tp["targets"])>1 else 0,
            "target_3":tp["targets"][2]["price"] if len(tp["targets"])>2 else 0,
            "risk_reward":tp["risk_reward"],
            "reasons":" | ".join(r["verdict"]["reasons"]),
            "warnings":" | ".join(r["verdict"]["warnings"]),
        })
    csv_path = os.path.join(DATA_DIR, f"analysis_{date_str}.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"  ✓ HTML → {html_path}")
    print(f"  ✓ CSV  → {csv_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
