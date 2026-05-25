"""
Momentum Riding Strategy — 자동화 모듈
매일 Fallen Angel 파이프라인과 함께 실행
결과: docs/momentum.html  +  data/momentum_YYYYMMDD.csv
"""

import os, json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ══════════════════════════════════════════════════════════
#  테마 유니버스
# ══════════════════════════════════════════════════════════

THEMES = [
    {"name": "AI 반도체",       "category": "메가트렌드",   "tier_base": "S",
     "etf": "SOXX",  "leaders": ["NVDA","AMD","AVGO"],    "followers": ["AMAT","MU","LRCX"]},
    {"name": "클라우드 SaaS",   "category": "메가트렌드",   "tier_base": "S",
     "etf": "WCLD",  "leaders": ["MSFT","CRM","NOW"],     "followers": ["SNOW","DDOG","MDB"]},
    {"name": "방위산업",         "category": "정치정책",     "tier_base": "A",
     "etf": "ITA",   "leaders": ["LMT","RTX","NOC"],      "followers": ["GD","HII","LDOS"]},
    {"name": "원자력 에너지",    "category": "메가트렌드",   "tier_base": "A",
     "etf": "NLR",   "leaders": ["CEG","VST","CCJ"],      "followers": ["UEC","NNE","SMR"]},
    {"name": "사이버보안",       "category": "메가트렌드",   "tier_base": "A",
     "etf": "HACK",  "leaders": ["CRWD","PANW","ZS"],     "followers": ["FTNT","S","OKTA"]},
    {"name": "바이오헬스케어",   "category": "메가트렌드",   "tier_base": "A",
     "etf": "XBI",   "leaders": ["LLY","REGN","VRTX"],   "followers": ["MRNA","ABBV","BIIB"]},
    {"name": "로봇 자동화",      "category": "메가트렌드",   "tier_base": "A",
     "etf": "ROBO",  "leaders": ["ISRG","TER","ONTO"],    "followers": ["BRKS","NOVT","ACMR"]},
    {"name": "에너지 원유",      "category": "글로벌매크로", "tier_base": "B",
     "etf": "XLE",   "leaders": ["XOM","CVX","COP"],      "followers": ["SLB","HAL","DVN"]},
    {"name": "금융",             "category": "글로벌매크로", "tier_base": "B",
     "etf": "XLF",   "leaders": ["JPM","GS","MS"],        "followers": ["BAC","C","WFC"]},
    {"name": "금 귀금속",        "category": "글로벌매크로", "tier_base": "B",
     "etf": "GDX",   "leaders": ["GOLD","NEM","AEM"],     "followers": ["AG","WPM","HL"]},
]

TIER_META = {
    "S": {"bg":"#c3f0d8","fg":"#1b4332","bd":"#6fcf97","label":"S등급 — 산업 구조 변화급","max_hold":60,"max_wt":40},
    "A": {"bg":"#c2e4f5","fg":"#0d2f45","bd":"#56b4d3","label":"A등급 — 정책 수혜 + 실적 연동","max_hold":15,"max_wt":30},
    "B": {"bg":"#fde8c8","fg":"#6b3a00","bd":"#f5a623","label":"B등급 — 이벤트 드리븐","max_hold":5,"max_wt":15},
    "C": {"bg":"#e8e9eb","fg":"#444",   "bd":"#b0b3b8","label":"C등급 — 진입 금지","max_hold":0,"max_wt":0},
}

STAGE_META = {
    "태동기": {"icon":"⏳","col":"#888",   "label":"관찰 대기"},
    "가속기": {"icon":"🚀","col":"#0F6E56","label":"최적 진입 구간"},
    "과열기": {"icon":"🔥","col":"#A32D2D","label":"진입 금지"},
    "조정기": {"icon":"↩️","col":"#854F0B","label":"S등급만 재진입 검토"},
    "소멸":   {"icon":"💀","col":"#aaa",   "label":"관심 해제"},
}


# ══════════════════════════════════════════════════════════
#  데이터 수집
# ══════════════════════════════════════════════════════════

def _rsi(series, n=14):
    d    = series.diff()
    gain = d.where(d > 0, 0.0).rolling(n).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(n).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def fetch_all(themes):
    tickers = {"SPY"}
    for t in themes:
        tickers.add(t["etf"])
        tickers.update(t["leaders"][:2])
        tickers.update(t["followers"][:1])

    raw    = yf.download(list(tickers), period="6mo", progress=False)
    close  = raw["Close"]  if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    volume = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
    return close, volume


# ══════════════════════════════════════════════════════════
#  PHASE 1 + 2 : 테마 분석
# ══════════════════════════════════════════════════════════

def analyze_theme(theme, close, volume):
    etf = theme["etf"]
    base = {
        "name": theme["name"], "category": theme["category"],
        "etf": etf, "tier": "C", "lifecycle": "소멸",
        "entry_eligible": False, "checklist_score": 0,
        "checklist": {}, "rsi": None,
        "return_5d": None, "return_20d": None, "return_60d": None,
        "volume_ratio": None, "excess_vs_spy": None,
        "stocks": [], "confidence": 1,
    }

    if etf not in close.columns:
        return base

    ec = close[etf].dropna()
    if len(ec) < 60:
        return base

    # ── 수익률 ──
    c   = float(ec.iloc[-1])
    p5  = float(ec.iloc[-6])  if len(ec) > 5  else c
    p20 = float(ec.iloc[-21]) if len(ec) > 20 else c
    p60 = float(ec.iloc[-61]) if len(ec) > 60 else c
    r5, r20, r60 = (c-p5)/p5, (c-p20)/p20, (c-p60)/p60

    # ── S&P500 초과 수익 ──
    excess = 0.0
    if "SPY" in close.columns:
        sp = close["SPY"].dropna()
        if len(sp) > 5:
            excess = r5 - (float(sp.iloc[-1]) - float(sp.iloc[-6])) / float(sp.iloc[-6])

    # ── RSI ──
    rsi_s = _rsi(ec)
    rsi   = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

    # ── 거래량 비율 ──
    vol_ratio = 1.0
    if etf in volume.columns:
        ev = volume[etf].dropna()
        if len(ev) >= 20:
            vol_ratio = float(ev.tail(5).mean()) / max(float(ev.tail(20).mean()), 1)

    # ── 체크리스트 ──
    leaders_60d = []
    for tk in theme["leaders"][:2]:
        if tk in close.columns:
            s = close[tk].dropna()
            if len(s) >= 61:
                leaders_60d.append((float(s.iloc[-1]) - float(s.iloc[-61])) / float(s.iloc[-61]))

    ck = {
        "earnings_proof":       sum(1 for p in leaders_60d if p > 0.05) >= 1,
        "policy_support":       theme["category"] in ("정치정책", "메가트렌드"),
        "global_sync":          r60 > 0.08 and excess > 0.01,
        "institutional_buying": vol_ratio > 1.3 and r5 > 0.01 and rsi > 50,
        "sector_etf_volume":    vol_ratio > 1.3,
        "clear_leader":         bool(leaders_60d) and max(leaders_60d, default=0) > 0.08,
        "multiple_sources":     vol_ratio > 1.5 or abs(r5) > 0.025,
    }
    ck_score = sum(ck.values())

    # ── 등급 ──
    tb = theme["tier_base"]
    if   ck_score >= 6: tier = "S" if tb in ("S","A") else "A"
    elif ck_score >= 4: tier = tb
    elif ck_score >= 2: tier = "B" if tb != "C" else "C"
    else:               tier = "C"

    # ── 라이프사이클 ──
    if tier == "C" or (r5 < -0.01 and r20 < -0.03):
        lc = "소멸"
    elif rsi >= 78 or (r20 > 0.28 and vol_ratio > 4):
        lc = "과열기"
    elif rsi >= 65 or (vol_ratio < 0.80 and r5 < 0.005):
        lc = "조정기"
    elif rsi >= 50 and r5 > 0.005 and vol_ratio >= 1.2:
        lc = "가속기"
    elif rsi >= 44 and r20 > -0.02:
        lc = "태동기"
    else:
        lc = "조정기"

    base.update({
        "tier": tier, "lifecycle": lc,
        "entry_eligible": lc == "가속기" and tier in ("S","A","B"),
        "checklist": ck, "checklist_score": ck_score,
        "rsi": round(rsi, 1),
        "return_5d":  round(r5*100, 2),
        "return_20d": round(r20*100, 2),
        "return_60d": round(r60*100, 2),
        "volume_ratio": round(vol_ratio, 2),
        "excess_vs_spy": round(excess*100, 2),
        "confidence": min(10, max(1, ck_score + (2 if lc == "가속기" else 0))),
    })

    # ── 종목 분석 ──
    for tk in theme["leaders"][:2] + theme["followers"][:1]:
        sd = _analyze_stock(tk, theme, close, volume, lc, tier)
        if sd:
            base["stocks"].append(sd)
    base["stocks"].sort(key=lambda x: -x["total_score"])

    return base


def _analyze_stock(ticker, theme, close, volume, lifecycle, tier):
    if ticker not in close.columns:
        return None
    s = close[ticker].dropna()
    if len(s) < 21:
        return None

    c   = float(s.iloc[-1])
    p5  = float(s.iloc[-6])  if len(s) > 5  else c
    p20 = float(s.iloc[-21]) if len(s) > 20 else c
    p60 = float(s.iloc[-61]) if len(s) > 60 else c
    r5, r20, r60 = (c-p5)/p5, (c-p20)/p20, (c-p60)/p60
    rsi_val = float(_rsi(s).iloc[-1]) if not pd.isna(_rsi(s).iloc[-1]) else 50.0

    tvr = 1.0
    if ticker in volume.columns:
        tv = volume[ticker].dropna()
        if len(tv) >= 20:
            tvr = float(tv.tail(5).mean()) / max(float(tv.tail(20).mean()), 1)

    # 펀더멘탈
    roe = dte = rg = fpe = None
    op = False
    try:
        info = yf.Ticker(ticker).info
        roe  = info.get("returnOnEquity")
        dte  = info.get("debtToEquity")
        rg   = info.get("revenueGrowth")
        fpe  = info.get("forwardPE")
        op   = (info.get("operatingMargins") or 0) > 0
    except Exception:
        pass

    fund_pass = all([
        roe and roe > 0.10,
        dte is None or dte < 200,
        rg  is None or rg > 0,
        op,
    ])

    # 그린 시그널
    green = sum([
        tvr > 2.0,
        r5  > 0.02,
        50 <= rsi_val <= 70,
        tvr > 1.5,
        fund_pass,
        r5 > 0 and r20 > 0,
    ])
    red = sum([
        r5 > 0.20,
        rsi_val > 75,
        rsi_val < 40,
        tvr < 0.5,
    ])

    role = "대장주" if ticker in theme["leaders"] else "후발주"

    # 스코어
    mom  = min(10, max(0, r5*200 + r20*50 + (rsi_val-50)*0.10 + (tvr-1)*3))
    rel  = 9 if role == "대장주" else 6
    sup  = min(10, max(0, (tvr-1)*5 + (rsi_val-50)*0.10))
    fun  = 8 if fund_pass else 4
    rr   = min(10, max(0, 6 + r5*50 - red*2))
    total = round(mom*0.30 + rel*0.25 + sup*0.20 + fun*0.15 + rr*0.10, 1)

    return {
        "ticker":        ticker,
        "role":          role,
        "price":         round(c, 2),
        "return_5d":     round(r5*100, 2),
        "return_20d":    round(r20*100, 2),
        "return_60d":    round(r60*100, 2),
        "rsi":           round(rsi_val, 1),
        "volume_ratio":  round(tvr, 2),
        "fund_pass":     fund_pass,
        "roe":           round(roe*100,1) if roe else None,
        "fwd_pe":        round(fpe,1) if fpe else None,
        "rev_growth":    round(rg*100,1) if rg else None,
        "green_signals": green,
        "red_signals":   red,
        "scores": {"momentum":round(mom,1),"relevance":rel,
                   "supply":round(sup,1),"fundamental":fun,
                   "risk_reward":round(rr,1),"total":total},
        "total_score":   total,
        "entry_eligible": total >= 7.0 and lifecycle == "가속기"
                          and tier in ("S","A","B") and red == 0,
    }


# ══════════════════════════════════════════════════════════
#  PHASE 4 : 진입 계획
# ══════════════════════════════════════════════════════════

def entry_plan(stock, capital, tier):
    p     = stock["price"]
    stop  = round(p * 0.93, 2)
    rps   = max(p - stop, p * 0.03)
    hold  = TIER_META[tier]["max_hold"]

    risk_shares = int(capital * 0.02 / rps)
    max_shares  = int(capital * TIER_META[tier]["max_wt"] / 100 / p)
    shares_1st  = min(risk_shares, max_shares) // 2   # 1차 = 50%
    amt_1st     = round(shares_1st * p, 2)

    return {
        "green":       stock["green_signals"],
        "red":         stock["red_signals"],
        "eligible":    stock["entry_eligible"],
        "type":        "1차 진입 (목표비중 50%)" if stock["entry_eligible"] else "진입 불가",
        "price":       p,
        "shares_1st":  shares_1st,
        "amount_1st":  amt_1st,
        "weight_1st":  round(amt_1st/capital*100, 1),
        "stop_loss":   stop,
        "tp1":         round(p * 1.10, 2),
        "tp2":         round(p * 1.20, 2),
        "tp3":         round(p * 1.30, 2),
        "max_hold":    hold,
        "trailing_activate": round(p * 1.15, 2),
        "trailing_stop_at":  round(p * 1.05, 2),
    }


# ══════════════════════════════════════════════════════════
#  HTML 생성
# ══════════════════════════════════════════════════════════

def _bdg(text, bg, fg, bd=None):
    border = f"border:1px solid {bd};" if bd else ""
    return (f'<span style="background:{bg};color:{fg};{border}'
            f'padding:2px 9px;border-radius:6px;font-size:12px;font-weight:600;white-space:nowrap">'
            f'{text}</span>')


def _checklist_html(ck):
    labels = {
        "earnings_proof":       "실적 증명 기업",
        "policy_support":       "정책/정부 지원",
        "global_sync":          "글로벌 동시 진행",
        "institutional_buying": "기관 매수 신호",
        "sector_etf_volume":    "ETF 거래량 증가",
        "clear_leader":         "대장주 명확",
        "multiple_sources":     "복수 뉴스/신호",
    }
    items = ""
    for k, v in ck.items():
        ic  = "✅" if v else "☐"
        col = "#0F6E56" if v else "#ccc"
        items += f'<span style="color:{col};font-size:12px;margin-right:10px">{ic} {labels.get(k,k)}</span>'
    return items


def _stock_card(s, tier, capital):
    ep    = entry_plan(s, capital, tier)
    tm    = TIER_META[tier]
    score = s["total_score"]
    sc    = "#0F6E56" if score >= 7 else "#854F0B" if score >= 5 else "#A32D2D"

    green_h = "".join(
        f'<div style="color:#0F6E56;font-size:12px">✅ {g}</div>'
        for g in [
            f"거래량 비율 {s['volume_ratio']:.1f}x (기준 2.0x)" if s["volume_ratio"]>2.0 else "",
            f"5일 수익률 +{s['return_5d']:.1f}% (기준 +2%)" if s["return_5d"]>2 else "",
            f"RSI {s['rsi']:.0f} — 모멘텀 구간 (50~70)" if 50<=s["rsi"]<=70 else "",
            "펀더멘탈 최소 기준 통과" if s["fund_pass"] else "",
        ] if g
    ) or '<span style="color:#aaa;font-size:12px">없음</span>'

    red_h = "".join(
        f'<div style="color:#A32D2D;font-size:12px">🚫 {r}</div>'
        for r in [
            f"5일 상승 {s['return_5d']:.1f}% — 과열 주의" if s["return_5d"]>20 else "",
            f"RSI {s['rsi']:.0f} — 과매수 구간" if s["rsi"]>75 else "",
            f"RSI {s['rsi']:.0f} — 모멘텀 부재" if s["rsi"]<40 else "",
            "거래량 급감 — 관심 이탈" if s["volume_ratio"]<0.5 else "",
        ] if r
    ) or '<span style="color:#aaa;font-size:12px">없음</span>'

    entry_color = "#0F6E56" if ep["eligible"] else "#A32D2D"
    entry_bg    = "#f0faf5" if ep["eligible"] else "#fff5f5"

    trade_h = ""
    if ep["eligible"]:
        trade_h = f"""
        <div style="background:{entry_bg};border:1px solid {entry_color};border-radius:8px;padding:12px;margin-top:10px">
          <div style="font-weight:600;color:{entry_color};margin-bottom:8px">
            🟢 진입 계획 — {ep['type']}
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:13px">
            <div><span style="color:#888;font-size:11px">1차 수량</span><br><b>{ep['shares_1st']}주</b></div>
            <div><span style="color:#888;font-size:11px">1차 금액</span><br><b>${ep['amount_1st']:,.0f}</b></div>
            <div><span style="color:#888;font-size:11px">포트폴리오 비중</span><br><b>{ep['weight_1st']}%</b></div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:12px;margin-top:10px">
            <div style="background:#FCEBEB;padding:6px 8px;border-radius:6px">
              <div style="color:#888;font-size:10px">손절 (-7%)</div>
              <b style="color:#A32D2D">${ep['stop_loss']:.2f}</b>
            </div>
            <div style="background:#E1F5EE;padding:6px 8px;border-radius:6px">
              <div style="color:#888;font-size:10px">목표1 (+10%)</div>
              <b style="color:#0F6E56">${ep['tp1']:.2f}</b>
            </div>
            <div style="background:#E1F5EE;padding:6px 8px;border-radius:6px">
              <div style="color:#888;font-size:10px">목표2 (+20%)</div>
              <b style="color:#0F6E56">${ep['tp2']:.2f}</b>
            </div>
            <div style="background:#E1F5EE;padding:6px 8px;border-radius:6px">
              <div style="color:#888;font-size:10px">목표3 (+30%)</div>
              <b style="color:#0F6E56">${ep['tp3']:.2f}</b>
            </div>
          </div>
          <div style="font-size:11px;color:#888;margin-top:8px">
            최대 보유 {ep['max_hold']}일 · +15% 도달 시 트레일링 스톱 +5% 발동
          </div>
        </div>"""
    else:
        reasons = []
        if ep["red"] > 0:
            reasons.append(f"RED 시그널 {ep['red']}개")
        if s["total_score"] < 7:
            reasons.append(f"종합점수 {score:.1f} (기준 7.0)")
        trade_h = f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:10px;margin-top:10px;color:#888;font-size:13px">
          진입 불가 — {' / '.join(reasons) or '조건 미충족'}
        </div>"""

    return f"""
    <div style="border:1px solid #e5e7eb;border-left:4px solid {sc};border-radius:10px;padding:16px;margin:0 0 12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px">
        <div>
          <span style="font-size:18px;font-weight:700;font-family:monospace">{s['ticker']}</span>
          <span style="font-size:12px;color:#888;margin-left:6px">{s['role']}</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <span style="font-size:22px;font-weight:700;color:{sc}">{score:.1f}</span>
          <span style="font-size:11px;color:#888">/ 10</span>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:12px">
        <div style="background:#f8f9fa;padding:6px 10px;border-radius:6px">
          <div style="font-size:10px;color:#888">현재가</div>
          <div style="font-weight:600">${s['price']:.2f}</div>
        </div>
        <div style="background:#f8f9fa;padding:6px 10px;border-radius:6px">
          <div style="font-size:10px;color:#888">5일</div>
          <div style="font-weight:600;color:{'#0F6E56' if s['return_5d']>=0 else '#A32D2D'}">{'+' if s['return_5d']>=0 else ''}{s['return_5d']:.1f}%</div>
        </div>
        <div style="background:#f8f9fa;padding:6px 10px;border-radius:6px">
          <div style="font-size:10px;color:#888">20일</div>
          <div style="font-weight:600;color:{'#0F6E56' if s['return_20d']>=0 else '#A32D2D'}">{'+' if s['return_20d']>=0 else ''}{s['return_20d']:.1f}%</div>
        </div>
        <div style="background:#f8f9fa;padding:6px 10px;border-radius:6px">
          <div style="font-size:10px;color:#888">RSI</div>
          <div style="font-weight:600;color:{'#A32D2D' if s['rsi']>75 else '#0F6E56' if 50<=s['rsi']<=70 else '#888'}">{s['rsi']:.0f}</div>
        </div>
        <div style="background:#f8f9fa;padding:6px 10px;border-radius:6px">
          <div style="font-size:10px;color:#888">거래량 비율</div>
          <div style="font-weight:600;color:{'#0F6E56' if s['volume_ratio']>=1.5 else '#888'}">{s['volume_ratio']:.1f}x</div>
        </div>
      </div>
      <details>
        <summary style="cursor:pointer;font-size:13px;font-weight:500;padding:4px 0;color:#555">진입 시그널 상세</summary>
        <div style="padding:8px 0;display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <div><div style="font-size:11px;font-weight:600;color:#0F6E56;margin-bottom:4px">🟢 GREEN ({s['green_signals']}개)</div>{green_h}</div>
          <div><div style="font-size:11px;font-weight:600;color:#A32D2D;margin-bottom:4px">🔴 RED ({s['red_signals']}개)</div>{red_h}</div>
        </div>
      </details>
      {trade_h}
    </div>"""


def _theme_section(t, capital):
    tm    = TIER_META[t["tier"]]
    sm    = STAGE_META.get(t["lifecycle"], STAGE_META["소멸"])
    score = t["checklist_score"]

    ck_bar = "".join(
        f'<span title="{k}" style="display:inline-block;width:26px;height:26px;border-radius:50%;'
        f'background:{"#6fcf97" if v else "#e5e7eb"};margin:2px;line-height:26px;text-align:center;font-size:12px">'
        f'{"✓" if v else ""}</span>'
        for k, v in t["checklist"].items()
    )

    stocks_h = "".join(_stock_card(s, t["tier"], capital) for s in t["stocks"])

    n_eligible = sum(1 for s in t["stocks"] if s["entry_eligible"])
    alert = ""
    if n_eligible > 0 and t["lifecycle"] == "가속기":
        alert = f"""<div style="background:#E1F5EE;border:1px solid #6fcf97;border-radius:8px;
                    padding:10px 14px;margin-bottom:14px;font-size:13px;font-weight:600;color:#0F6E56">
                    🚀 {n_eligible}개 종목 진입 가능 — 가속기 구간, 1차 진입 검토</div>"""

    return f"""
<div style="border:1px solid #e5e7eb;border-top:4px solid {tm['bd']};border-radius:10px;padding:20px;margin:0 0 20px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:14px">
    <div>
      <span style="font-size:18px;font-weight:700">{t['name']}</span>
      <span style="font-size:12px;color:#888;margin-left:8px">{t['etf']} · {t['category']}</span>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap">
      {_bdg(tm['label'], tm['bg'], tm['fg'], tm['bd'])}
      {_bdg(f"{sm['icon']} {t['lifecycle']} — {sm['label']}", '#f8f9fa', sm['col'])}
      {_bdg(f"신뢰도 {t['confidence']}/10", '#f8f9fa', '#555')}
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px">
    <div style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center">
      <div style="font-size:10px;color:#888;text-transform:uppercase">ETF RSI</div>
      <div style="font-size:18px;font-weight:700;color:{'#A32D2D' if (t['rsi'] or 0)>75 else '#0F6E56' if 50<=(t['rsi'] or 0)<=70 else '#555'}">{t['rsi'] or '—'}</div>
    </div>
    <div style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center">
      <div style="font-size:10px;color:#888">5일 수익률</div>
      <div style="font-size:18px;font-weight:700;color:{'#0F6E56' if (t['return_5d'] or 0)>=0 else '#A32D2D'}">{('+' if (t['return_5d'] or 0)>=0 else '')}{t['return_5d'] or '—'}%</div>
    </div>
    <div style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center">
      <div style="font-size:10px;color:#888">20일 수익률</div>
      <div style="font-size:18px;font-weight:700;color:{'#0F6E56' if (t['return_20d'] or 0)>=0 else '#A32D2D'}">{('+' if (t['return_20d'] or 0)>=0 else '')}{t['return_20d'] or '—'}%</div>
    </div>
    <div style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center">
      <div style="font-size:10px;color:#888">거래량 비율</div>
      <div style="font-size:18px;font-weight:700;color:{'#0F6E56' if (t['volume_ratio'] or 0)>=1.3 else '#888'}">{t['volume_ratio'] or '—'}x</div>
    </div>
    <div style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center">
      <div style="font-size:10px;color:#888">SPY 초과</div>
      <div style="font-size:18px;font-weight:700;color:{'#0F6E56' if (t['excess_vs_spy'] or 0)>0 else '#A32D2D'}">{('+' if (t['excess_vs_spy'] or 0)>=0 else '')}{t['excess_vs_spy'] or '—'}%</div>
    </div>
  </div>

  <div style="margin-bottom:14px">
    <div style="font-size:11px;color:#888;margin-bottom:4px">체크리스트 {score}/7</div>
    {ck_bar}
    <div style="margin-top:4px;font-size:11px;color:#888">{_checklist_html(t['checklist'])}</div>
  </div>

  {alert}
  {stocks_h or '<p style="color:#aaa;font-size:13px">종목 데이터 없음</p>'}
</div>"""


def generate_html(themes_sorted, capital, date_str):
    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 요약
    actionable_themes  = [t for t in themes_sorted if t["entry_eligible"]]
    actionable_stocks  = sum(1 for t in themes_sorted for s in t["stocks"] if s["entry_eligible"])
    heating            = [t for t in themes_sorted if t["lifecycle"] == "과열기"]

    # 테마 순서: S→A→B→C, 가속기 우선
    def sort_key(t):
        tier_o  = {"S":0,"A":1,"B":2,"C":3}.get(t["tier"], 4)
        stage_o = {"가속기":0,"태동기":1,"조정기":2,"과열기":3,"소멸":4}.get(t["lifecycle"], 5)
        return (tier_o, stage_o, -(t["return_5d"] or 0))

    themes_sorted = sorted(themes_sorted, key=sort_key)

    # 테마 요약 테이블
    tbl_rows = ""
    for t in themes_sorted:
        tm = TIER_META[t["tier"]]
        sm = STAGE_META.get(t["lifecycle"], STAGE_META["소멸"])
        r5_col  = "#0F6E56" if (t["return_5d"] or 0) >= 0 else "#A32D2D"
        exc_col = "#0F6E56" if (t["excess_vs_spy"] or 0) > 0 else "#A32D2D"
        tbl_rows += f"""<tr>
          <td><b>{t['name']}</b> <span style="color:#aaa;font-size:11px">{t['etf']}</span></td>
          <td>{_bdg(t['tier']+'등급', tm['bg'], tm['fg'], tm['bd'])}</td>
          <td>{_bdg(sm['icon']+' '+t['lifecycle'], '#f8f9fa', sm['col'])}</td>
          <td style="text-align:right;color:#555">{t['rsi'] or '—'}</td>
          <td style="text-align:right;color:{r5_col};font-weight:600">{('+' if (t['return_5d'] or 0)>=0 else '')}{t['return_5d'] or '—'}%</td>
          <td style="text-align:right;color:{exc_col}">{('+' if (t['excess_vs_spy'] or 0)>=0 else '')}{t['excess_vs_spy'] or '—'}%</td>
          <td style="text-align:right">{t['volume_ratio'] or '—'}x</td>
          <td style="text-align:right">{t['checklist_score']}/7</td>
          <td style="text-align:right">{t['confidence']}/10</td>
        </tr>"""

    # 카드 (C등급·소멸 제외)
    cards_h = "".join(
        _theme_section(t, capital)
        for t in themes_sorted
        if not (t["tier"] == "C" and t["lifecycle"] == "소멸")
    )

    alert_bar = ""
    if actionable_stocks > 0:
        alert_bar = f"""<div style="background:#E1F5EE;border:2px solid #6fcf97;border-radius:12px;
            padding:14px 18px;margin:0 0 24px;font-size:14px;font-weight:600;color:#0F6E56">
            🚀 오늘 진입 가능 종목 <span style="font-size:20px">{actionable_stocks}개</span>
             · {len(actionable_themes)}개 테마가 가속기 구간입니다.
        </div>"""
    elif heating:
        alert_bar = f"""<div style="background:#FCEBEB;border:2px solid #e57373;border-radius:12px;
            padding:14px 18px;margin:0 0 24px;font-size:14px;font-weight:600;color:#A32D2D">
            🔥 {len(heating)}개 테마 과열 구간 — 신규 진입 자제, 보유 시 이탈 준비
        </div>"""
    else:
        alert_bar = """<div style="background:#f8f9fa;border:1px solid #e5e7eb;border-radius:12px;
            padding:14px 18px;margin:0 0 24px;font-size:14px;color:#888">
            ⏳ 현재 가속기 진입 가능 종목 없음 — 관찰 유지, 시장 대기
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum Riding — {date_fmt}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI",sans-serif;max-width:1050px;margin:0 auto;
     padding:24px 16px;color:#1a1a1a;line-height:1.6;background:#fff}}
h1{{font-size:22px;font-weight:700}} h2{{font-size:16px;font-weight:600;margin:28px 0 14px;
  padding:0 0 8px;border-bottom:2px solid #e5e7eb}}
.back{{font-size:13px;color:#888;text-decoration:none;display:block;margin:0 0 20px}}
.back:hover{{color:#185FA5}}
.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 24px}}
.stat{{background:#f8f9fa;padding:12px 14px;border-radius:10px}}
.sl{{font-size:11px;text-transform:uppercase;color:#888;letter-spacing:.4px}}
.sv{{font-size:22px;font-weight:700;margin:2px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:7px 10px;border-bottom:2px solid #e5e7eb;color:#888;font-weight:500;font-size:11px;white-space:nowrap}}
td{{padding:6px 10px;border-bottom:1px solid #f5f5f5;white-space:nowrap}}
details summary{{cursor:pointer;list-style:none;user-select:none}}
details summary::-webkit-details-marker{{display:none}}
@media(max-width:600px){{.stat-grid{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>

<a class="back" href="index.html">← Fallen Angel 리포트로 돌아가기</a>
<h1>⚡ Momentum Riding Strategy</h1>
<p style="color:#888;font-size:13px;margin:4px 0 20px">
  {date_fmt} · 업데이트: {run_time} KST · 자본금 ${capital:,.0f}
  &nbsp;·&nbsp; <b>핵심 원칙: 가는 길에 탑승 → 과열 전에 하차</b>
</p>

{alert_bar}

<div class="stat-grid">
  <div class="stat"><div class="sl">스캔 테마</div><div class="sv">{len(themes_sorted)}개</div></div>
  <div class="stat"><div class="sl">🚀 가속기 테마</div><div class="sv" style="color:#0F6E56">{len(actionable_themes)}개</div></div>
  <div class="stat"><div class="sl">진입 가능 종목</div><div class="sv" style="color:#0F6E56">{actionable_stocks}개</div></div>
  <div class="stat"><div class="sl">🔥 과열 주의</div><div class="sv" style="color:#A32D2D">{len(heating)}개</div></div>
</div>

<h2>테마 현황 요약</h2>
<div style="overflow-x:auto;margin:0 0 24px">
<table>
  <thead><tr>
    <th>테마</th><th>등급</th><th>단계</th>
    <th style="text-align:right">RSI</th>
    <th style="text-align:right">5일 수익</th>
    <th style="text-align:right">SPY 초과</th>
    <th style="text-align:right">거래량비율</th>
    <th style="text-align:right">체크리스트</th>
    <th style="text-align:right">신뢰도</th>
  </tr></thead>
  <tbody>{tbl_rows}</tbody>
</table>
</div>

<h2>테마별 상세 분석</h2>
{cards_h}

<div style="font-size:12px;color:#888;margin:28px 0 0;padding:14px;background:#f8f9fa;border-radius:10px">
  본 리포트는 교육·분석 목적이며 투자 권유가 아닙니다. 최종 투자 판단과 책임은 사용자에게 있습니다.
</div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════

def run(capital, docs_dir, data_dir):
    print("  [Momentum] 테마 스캔 시작 …")
    date_str = datetime.now().strftime("%Y%m%d")

    try:
        close, volume = fetch_all(THEMES)
    except Exception as e:
        print(f"  [Momentum] 데이터 로드 실패: {e}"); return

    analyzed = []
    for theme in THEMES:
        try:
            analyzed.append(analyze_theme(theme, close, volume))
        except Exception as e:
            print(f"  [Momentum] {theme['name']} 실패: {e}")

    # HTML
    html      = generate_html(analyzed, capital, date_str)
    html_path = os.path.join(docs_dir, "momentum.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # CSV
    rows = []
    for t in analyzed:
        for s in t["stocks"]:
            rows.append({
                "theme": t["name"], "tier": t["tier"], "lifecycle": t["lifecycle"],
                "etf_rsi": t["rsi"], "etf_5d": t["return_5d"],
                "etf_volume_ratio": t["volume_ratio"], "checklist_score": t["checklist_score"],
                "ticker": s["ticker"], "role": s["role"], "price": s["price"],
                "return_5d": s["return_5d"], "return_20d": s["return_20d"],
                "rsi": s["rsi"], "vol_ratio": s["volume_ratio"],
                "total_score": s["total_score"], "entry_eligible": s["entry_eligible"],
            })
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(data_dir, f"momentum_{date_str}.csv"), index=False
        )

    n_act = sum(1 for t in analyzed for s in t["stocks"] if s["entry_eligible"])
    print(f"  [Momentum] 완료: {len(analyzed)}개 테마, 진입 가능 {n_act}개 종목")
    print(f"  ✓ Momentum → {html_path}")
