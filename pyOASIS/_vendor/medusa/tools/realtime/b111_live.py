#!/usr/bin/env python3
"""MEDUSA B111 live sidecar, BETA.

Reads the existing C++ live HTTP API (no GeoRinex, no RTCM re-decoding),
reuses MEDUSA B111's Bernese-inspired QC, and publishes an atomic JSON file.
Never changes the GNSS backend or historical B111/D111 source.

Only records observed dual-frequency phase AND code from the same epoch.
GLONASS FDMA is deliberately skipped unless the API supplies channel/frequency.
Anomaly scores are *screening indicators*, not verified ionospheric detections.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd

from tools.historical.b111.combinations import (
    FrequencyPair, geometry_free_phase, bernese_l5_mw_cycles,
)
from tools.historical.b111.rinex_driver import (
    B111RinexCombination, run_b111_rinex_qc,
)
from tools.historical.b111.run_real_all_satellites import choose_combination
from tools.historical.b111.screening import bernese_offset

LOGGER = logging.getLogger("medusa.b111.live")
C = 299792458.0
# Exact nominal center frequencies; do not guess GLONASS FDMA channel.
FREQ_HZ = {
    "G": {"1":1575.42e6,"2":1227.60e6,"5":1176.45e6},
    "E": {"1":1575.42e6,"5":1176.45e6,"7":1207.14e6,
          "8":1191.795e6,"6":1278.75e6},
    "C": {"1":1575.42e6,"2":1561.098e6,"5":1176.45e6,
          "6":1268.52e6,"7":1207.14e6,"8":1191.795e6},
    "R": {"3":1202.025e6,"4":1600.995e6,"6":1248.06e6},
}
CONSTELLATIONS = {"GPS":"G","GLONASS":"R","GALILEO":"E", "BEIDOU":"C", "BDS":"C"}
COLORS = {"G":"#ef4444", "R":"#3b82f6", "E":"#22c55e", "C":"#eab308"}
# The B111 hard thresholds are from the existing historical runner.
B111_CONFIG = dict(gaparc=0.05, minobs=10, rmsl5=0.60, slpmin=1.0,
                   slpout=5.0, rmsl4=0.01, gapl4=180/3600,
                   minl4=10, ifxslp=1, offset_func=bernese_offset)


def finite(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def obs_freq(sat, signal_id, obs):
    """Prefer frequency explicitly supplied by the RTCM decoder.

    Some APIs expose 1C, others L1C. We only infer non-GLONASS
    system frequencies from a known RINEX frequency digit.
    """
    for key in ("carrier_frequency_hz", "frequency_hz", "signal_frequency_hz"):
        v = finite(obs.get(key))
        if v and v > 1e8:
            return v
    if sat[0] == "R" and signal_id[0] in "12":
        k = None
        for key in ("glonass_frequency_channel", "frequency_channel", "glo_fcn", "fcn"):
            if obs.get(key) is not None:
                try:
                    k = int(obs[key]); break
                except (TypeError, ValueError):
                    pass
        if k is None or not -7 <= k <= 6:
            return None
        return (1602.0 + k * 0.5625)*1e6 if signal_id[0] == "1" else (1246.0 + k * 0.4375)*1e6
    return FREQ_HZ.get(sat[0], {}).get(signal_id[0])


def canonical_signal(signal_id):
    s = str(signal_id or "").strip().upper()
    # RTKLIB signals may be 'L1C', '1C', or 'C1C'.
    if len(s) == 3 and s[0] in "LCSGREC" and s[1].isdigit():
        s = s[1:]
    if re.fullmatch(r"[1-9][A-Z]", s):
        return s
    return None


def collect_epoch_rows(data):
    """Return (station mountpoint, valid standardized epoch/observation records).

    Does not synthesize absent observations or convert Hz to cycles.
    """
    mountpoint = str(data.get("mountpoint") or "").strip()
    rows = []
    for epoch in data.get("epochs", []):
        try:
            ms = int(epoch["utc_ms"])
        except (TypeError, ValueError, KeyError):
            continue
        for obs in epoch.get("observations", []):
            sat = str(obs.get("satellite") or "").strip().upper()
            if not re.fullmatch(r"[GREC]\d{1,3}", sat):
                continue
            sys = CONSTELLATIONS.get(str(obs.get("constellation") or "").upper())
            if sys and sys != sat[0]:
                continue
            sig = canonical_signal(obs.get("signal"))
            phase = finite(obs.get("carrier_phase_cycles"))
            code = finite(obs.get("pseudorange_m"))
            if not sig or phase is None or code is None or phase == 0 or code <= 0:
                continue
            freq = obs_freq(sat, sig, obs)
            if freq is None:
                continue
            rows.append((ms, sat, sig, phase, code, freq))
    return mountpoint, rows


def build_satellite_combinations(records):
    """Join code and phase signals by exact epoch; no frequency interpolation."""
    # records: (utc_ms, sat, signal, phase, code, frequency)
    times = sorted({r[0] for r in records})
    if len(times) < 12:
        return []
    signals = sorted({r[2] for r in records})
    lookup = {(r[0], r[2]): r for r in records}
    if len(signals) < 2:
        return []
    sat = records[0][1]
    candidates = []
    for i, s1 in enumerate(signals):
        for s2 in signals[i+1:]:
            if s1[0] == s2[0]:
                continue
            # Priority policy in choose_combination selects scientific
            # frequency pairs; candidates never cross sat or epoch.
            n = len(times)
            p1=np.full(n,np.nan); p2=np.full(n,np.nan)
            c1=np.full(n,np.nan); c2=np.full(n,np.nan)
            f1s=[]; f2s=[]
            for j, t in enumerate(times):
                r1, r2 = lookup.get((t,s1)), lookup.get((t,s2))
                if r1 is None or r2 is None:
                    continue
                p1[j], c1[j], p2[j], c2[j] = r1[3],r1[4],r2[3],r2[4]
                f1s.append(r1[5]); f2s.append(r2[5])
            if len(f1s) < B111_CONFIG["minobs"]:
                continue
            # Frequency/channel changing on the same satellite -> do not
            # generate a single ambiguous arc across incompatible carriers.
            f1=float(np.median(f1s)); f2=float(np.median(f2s))
            if (np.max(np.abs(np.asarray(f1s)-f1)) > 1 or
                np.max(np.abs(np.asarray(f2s)-f2)) > 1 or abs(f1-f2) < 1e5):
                continue
            valid=np.isfinite(p1)&np.isfinite(p2)&np.isfinite(c1)&np.isfinite(c2)
            pair=FrequencyPair(f1=f1, f2=f2)
            l4=geometry_free_phase(p1,p2,pair)
            mw=bernese_l5_mw_cycles(p1,p2,c1,c2,pair)
            candidates.append(B111RinexCombination(
                satellite=sat,
                phase1="L"+s1,phase2="L"+s2,
                code1="C"+s1, code2="C"+s2,
                f1_hz=f1,f2_hz=f2,
                times=np.asarray(times,dtype="datetime64[ms]"),
                phase1_cycles=p1, phase2_cycles=p2,
                code1_m=c1,code2_m=c2,
                l4_m=np.asarray(l4,dtype=float),
                # Signs MUST agree on first-order ionosphere:
                # L_GF=L1-L2; P_GF=P2-P1
                code_gf_m=c2-c1,
                l5_cycles=np.asarray(mw,dtype=float),
                valid=valid,
            ))
    return candidates


def detect_residual_anomalies(points, times_ms, threshold_min_m=0.8):
    """Causal rolling median/MAD anomaly screening, independent per arc.

    This is a beta fluctuation screening index, not a validated storm alarm.
    Insist on three consecutive abnormal epochs; missing epochs reset.
    """
    if len(points)<12:
        return [], [None]*len(points)
    s=pd.Series(points,index=pd.to_datetime(times_ms,unit="ms",utc=True),dtype=float)
    baseline=s.rolling("600s",min_periods=8).median().shift(1)
    resid=s-baseline
    mad=(resid.abs().rolling("1800s",min_periods=12).median().shift(1))
    sigma=1.4826*mad
    threshold=np.maximum(threshold_min_m,6.0*sigma.fillna(np.inf).to_numpy())
    residual_list=[None if not math.isfinite(x) else float(x) for x in resid.to_numpy()]
    anomalies=[]; run=0
    for i, v in enumerate(residual_list):
        if v is not None and math.isfinite(threshold[i]) and abs(v)>threshold[i]:
            run+=1
        else:
            run=0
        if run==3:
            anomalies.append({"utc_ms":int(times_ms[i]),"residual_m":round(float(v),4),
                "threshold_m":round(float(threshold[i]),4),"kind":"GF_RESIDUAL_PERSISTENT",
                "provisional":True})
    return anomalies,residual_list


def process_satellite(records):
    sat=records[0][1]
    combos=build_satellite_combinations(records)
    combo=choose_combination(combos,sat[0]) if combos else None
    if combo is None:
        return None, "No eligible dual-frequency phase/code pair"
    try:
        result=run_b111_rinex_qc(combo, **B111_CONFIG)
    except (ValueError,ArithmeticError,FloatingPointError,IndexError) as exc:
        return None, f"QC rejected {sat}: {type(exc).__name__}: {exc}"
    up=result.final_updarc
    leveled=np.asarray(result.leveling_result.leveled_gf_m,dtype=float)
    flags=np.asarray(up.flags,dtype=np.int64)
    # The QC driver returns the mapping between its processed
    # observations and the original combination epochs.
    #
    # This mapping belongs to B111RinexQcResult, not UpdarcResult.
    src = np.asarray(result.source_indices, dtype=int)

    all_t = np.asarray(
        combo.times,
        dtype="datetime64[ms]"
    )

    n = len(src)

    if not (
        len(leveled) == n
        and len(flags) == n
        and len(up.observations) == n
    ):
        return None, (
            f"QC index length mismatch for {sat}: "
            f"source={n}, leveled={len(leveled)}, "
            f"flags={len(flags)}, observations={len(up.observations)}"
        )

    if np.any(src < 0) or np.any(src >= len(all_t)):
        return None, f"Invalid source indices for {sat}"

    t_ms = all_t[src].astype("int64")

    usable=((flags&1)==0)&np.isfinite(leveled)
    per_arc=[]; anomalies=[]; all_points=[]; all_resid=[]; qc_events=[]
    # Native B111 CHKWUB / CHKL4P OUTLIER: LINFLG bit 0.
    # Never infer outliers from graph gaps or provisional residual screening.
    for rejected_idx in np.flatnonzero((flags & 1) != 0):
        qc_events.append({"utc_ms":int(t_ms[rejected_idx]),"satellite":sat,
                          "kind":"B111_OUTLIER","provisional":False,
                          "flag":int(flags[rejected_idx])})

    for ai,arc in enumerate(up.arcs):
        for si in (arc.slip_indices or []):
            si=int(si)
            if 0 <= si < len(t_ms):
                qc_events.append({"utc_ms":int(t_ms[si]),"satellite":sat,
                                  "arc":ai+1,"kind":"B111_CYCLE_SLIP",
                                  "provisional":False})
        a,b=int(arc.start),int(arc.end)+1
        ix=np.arange(a,b,dtype=int)
        ix=ix[usable[ix]]
        if len(ix)<2: continue
        t=t_ms[ix]; v=leveled[ix]
        # Never bridge discontinuities, missing epochs, or separately
        # levelled UPDARC arcs.
        dx=np.diff(t)
        positive=dx[dx>0]
        # 15 seconds is common but derived from the observed cadence.
        cadence=float(np.median(positive)) if positive.size else 15000.0
        breaks=np.r_[0,np.where((dx>1.5*cadence)|(dx<=0))[0]+1,len(ix)]
        for seg in range(len(breaks)-1):
            lo,hi=breaks[seg],breaks[seg+1]
            si=ix[lo:hi]
            if len(si)<2:continue
            st=t_ms[si]; sv=leveled[si]
            events,residual=detect_residual_anomalies(sv,st)
            for ev in events:
                ev.update({"satellite":sat,"arc":ai+1})
                anomalies.append(ev)
            all_points.append({"arc":ai+1,"t":st.tolist(),"y":np.around(sv,5).tolist()})
            all_resid.append({"arc":ai+1,"t":st.tolist(),"y":residual})
    if not all_points and not qc_events:
        return None,"QC completed but no valid leveled arcs or QC events"
    return {"satellite":sat,"system":sat[0],"color":COLORS[sat[0]],
        "signals":[combo.phase1,combo.phase2,combo.code1,combo.code2],
        "points":all_points,"residuals":all_resid,
        "frequencies_hz":[float(combo.f1_hz),float(combo.f2_hz)],
        "anomalies":anomalies,"qc_events":qc_events,"arc_count":len(up.arcs),
        "valid_epochs":int(np.count_nonzero(usable)),
        # Internal, observed dual-frequency epochs (not generated epochs).
        # The session cache uses a >GAPARC boundary to distinguish a closed
        # historical-B111 temporal block from an open/provisional block.
        "_pair_epoch_ms": all_t[np.asarray(combo.valid, dtype=bool)].astype('int64').tolist()},None


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def build_snapshot(data,rows,hist,now_ms):
    by_sat=defaultdict(list)
    for r in hist.values():
        by_sat[r[1]].append(r)
    satellites=[];failures={};anomalies=[];qc_events=[]
    for sat, records in sorted(by_sat.items()):
        try:
            result, reason = process_satellite(records)
            if result:
                satellites.append(result)
                anomalies.extend(result["anomalies"])
                qc_events.extend(result["qc_events"])
            else:
                failures[sat] = reason
        except Exception as exc:
            # An unexpected satellite failure must never prevent the
            # station's other satellites from being published.
            LOGGER.exception("B111 QC failed for %s", sat)
            failures[sat] = f"Unexpected QC error: {type(exc).__name__}: {exc}"
    return {"version":1,"processing":"B111_LIVE_BETA",
        "published_utc":datetime.now(timezone.utc).isoformat(),
        "mountpoint":data.get("mountpoint"),
        "receiver_epochs_in_window":len({r[0] for r in hist.values()}),
        "satellites":satellites,
        "anomalies":sorted(anomalies,key=lambda x:x["utc_ms"]),
        "qc_events":sorted(qc_events,key=lambda x:x["utc_ms"]),
        "skipped_satellites":failures,
        "notice":"GF fluctuation flags are provisional screening, not verified ionospheric anomalies; uncorrected DCBs remain."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--api",default="http://127.0.0.1:8081/api/v1/gnss/live")
    p.add_argument("--output",default="data/realtime/qc/b111_live.json")
    p.add_argument("--poll-seconds",type=float,default=15.0)
    p.add_argument("--window-minutes",type=float,default=30.0)
    p.add_argument("--once",action="store_true")
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    output=Path(args.output)
    hist={};active_mount=None
    while True:
        started=time.monotonic()
        try:
            with urlopen(args.api,timeout=12) as resp:
                data=json.load(resp)
            mount, rows=collect_epoch_rows(data)
            if not mount:
                raise ValueError("Live API has no active mountpoint")
            if mount!=active_mount:
                LOGGER.info("Station switch: %s -> %s; clearing QC buffer",active_mount,mount)
                hist.clear(); active_mount=mount
            for r in rows:
                hist[(r[0],r[1],r[2])]=r
            newest=max((r[0] for r in hist.values()), default=0)
            # Use GNSS measurement UTC for retention, not computer wall clock.
            cutoff=newest-int(args.window_minutes*60_000)
            hist={k:v for k,v in hist.items() if v[0]>=cutoff}
            snapshot=build_snapshot(data,rows,hist,newest)
            atomic_json(output,snapshot)
            LOGGER.info("%s: receiver epochs=%d, QC satellites=%d, anomalies=%d, skipped=%d",
                mount,snapshot["receiver_epochs_in_window"],len(snapshot["satellites"]),
                len(snapshot["anomalies"]),len(snapshot["skipped_satellites"]))
        except Exception:
            LOGGER.exception("Live B111 update failed; previous snapshot retained")
            if args.once: raise
        if args.once: break
        time.sleep(max(1.0,args.poll_seconds-(time.monotonic()-started)))

if __name__=="__main__":
    main()
