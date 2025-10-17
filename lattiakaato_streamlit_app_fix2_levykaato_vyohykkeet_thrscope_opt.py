# -*- coding: utf-8 -*-
"""
Lattiakaato PRO2 – FIX2 — LEVYKAATO + Kynnys (puoli & D_max) + KAKSI VYÖHYKETTÄ + KULMAN OPTIMOINTI

Uutta tässä versiossa:
- LEVYKAATO (planar outside slope) ulkoalueelle — nyt jopa KAKSI vyöhykettä (A/B) jakolinjan perusteella.
- Kullekin vyöhykkeelle oma kohdesuihku, suunta:
    • Automaattinen (kohti suihkua),
    • Manuaalinen kulma, tai
    • Optimointi (etsii kulman, joka minimoi reunan jatkuvuuden vaatiman c-kertoimen → alin mahdollinen taso).
- Kynnys: yksipuolinen vaikutus (vasen/oikea) ja enimmäisetäisyys D_max.
- Vyöhykerajoitus: levykaato-vyöhykkeessä voidaan haluttaessa ohittaa muiden suihkujen ja pistekaivojen ehdot (pitää pinnan siistinä).

Huomioi: Suihkun ulkopuolella levykaatotaso h(x,y)=g·[x,y]^T + c valitaan siten, että
  (1) |g| = k_o_min,
  (2) h >= step + h_edge_min KAIKISSA valitun suihkun reunan pisteissä (jatkuvuus/step-reuna),
  (3) kulma voidaan valita automaattisesti, manuaalisesti tai optimoida minimiin.
"""
import io, math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Geometria ----------------

def distance_point_to_rect(px, py, rect):
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    return math.hypot(px - cx, py - cy)

def project_point_to_rect(px, py, rect):
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    d = math.hypot(px - cx, py - cy)
    return cx, cy, d

def inside_rect(px, py, rect):
    x0, y0, w, h = rect
    return (x0 <= px <= x0 + w) and (y0 <= py <= y0 + h)

def signed_side_of_line(px, py, line):
    x0, y0, x1, y1 = line
    return (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)

def distance_point_to_segment(px, py, x0, y0, x1, y1):
    vx, vy = x1 - x0, y1 - y0
    wx, wy = px - x0, py - y0
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return math.hypot(px - x0, py - y0)
    t = (wx*vx + wy*vy) / seg_len2
    t = max(0.0, min(1.0, t))
    projx, projy = x0 + t*vx, y0 + t*vy
    return math.hypot(px - projx, py - projy)

# ---------------- Data ----------------
@dataclass
class Shower:
    rect: Tuple[float, float, float, float]  # x0,y0,w,h
    drain_x: float
    drain_y: float

@dataclass
class SplitLine:
    line: Tuple[float, float, float, float]  # x0,y0,x1,y1
    left_target_type: Optional[str]          # 'shower' | 'drain' | None
    left_target_index: Optional[int]
    right_target_type: Optional[str]
    right_target_index: Optional[int]

@dataclass
class PlanarZoneCfg:
    enabled: bool
    shower_idx: Optional[int]
    mode: str                 # 'auto' | 'manual' | 'opt'
    dir_deg: Optional[float]  # used if mode=='manual'
    zone_is_left: bool        # this zone applies to left (True) or right (False) side of zone splitline
    enforce_scope: bool       # if True: ignore other showers/drains in this zone (vyöhykerajoitus)

@dataclass
class Config:
    L: float; W: float; grid: float
    showers: List[Shower]
    drains: List[Tuple[float, float]]  # extra drains
    step_mm: float                     # korkoero suihku ↔ muu
    k_s_min: float; k_o_min: float     # 1/N
    # Kynnys: aina AWAY (yläraja)
    threshold: Optional[Tuple[float, float, float, float]]
    thr_H_mm: float
    thr_k_min: float                   # 1/N
    # Kynnys laajennukset
    thr_one_sided: bool
    thr_side_is_left: bool
    thr_D_max: Optional[float]
    # Kaatoalueiden rajaviivat (ohjaus)
    split_lines: List[SplitLine]
    # Kerrokset (valukorko)
    layers_m: float
    # Levykaato vyöhykkeet
    use_zone_split: bool
    zone_split_line: Optional[Tuple[float,float,float,float]]
    zoneA: PlanarZoneCfg
    zoneB: PlanarZoneCfg
    # Optimoinnin asetukset
    opt_angle_step_deg: float

# ------------- Rajaviivaohjaus -------------

def select_targets_by_splitlines(x, y, cfg: Config):
    sel_s = None
    sel_d = None
    for sl in cfg.split_lines:
        side = signed_side_of_line(x, y, sl.line)
        if side >= 0:  # vasen
            if sl.left_target_type == 'shower':
                sel_s, sel_d = sl.left_target_index, None
            elif sl.left_target_type == 'drain':
                sel_s, sel_d = None, sl.left_target_index
        else:  # oikea
            if sl.right_target_type == 'shower':
                sel_s, sel_d = sl.right_target_index, None
            elif sl.right_target_type == 'drain':
                sel_s, sel_d = None, sl.right_target_index
    return sel_s, sel_d

# ------------- Apurit levykaadolle -------------

def sample_rect_perimeter(rect, n_per_edge=80):
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    pts = []
    for i in range(n_per_edge):  # bottom
        t = i/(n_per_edge-1)
        pts.append((x0 + t*(x1-x0), y0))
    for i in range(1, n_per_edge):  # right
        t = i/(n_per_edge-1)
        pts.append((x1, y0 + t*(y1-y0)))
    for i in range(1, n_per_edge):  # top
        t = i/(n_per_edge-1)
        pts.append((x1 - t*(x1-x0), y1))
    for i in range(1, n_per_edge-1):  # left
        t = i/(n_per_edge-1)
        pts.append((x0, y1 - t*(y1-y0)))
    return pts


def compute_planar_g_c_for_shower(cfg: Config, sh: Shower, mode: str, dir_deg: Optional[float]):
    """Palauta (g,c, angle_used_deg, c_value) levykaatotasolle kohti suihkua.
    - mode='auto'|'manual'|'opt'
    - manual: käyttää dir_deg
    - auto: suunta huoneen keskipisteestä lähimmälle suihkureunan pisteelle
    - opt: etsii kulman [-180,180] step=cfg.opt_angle_step_deg, joka minimoi c:n
    """
    def g_c_from_angle(theta_deg: float):
        ang = math.radians(theta_deg)
        u = np.array([math.cos(ang), math.sin(ang)], dtype=float)
        if np.linalg.norm(u) < 1e-12:
            u = np.array([1.0, 0.0])
        g = -cfg.k_o_min * u  # laskee kohti suihkua
        step_m = cfg.step_mm/1000.0
        c_candidates = []
        for (rx, ry) in sample_rect_perimeter(sh.rect, n_per_edge=96):
            r_edge = math.hypot(rx - sh.drain_x, ry - sh.drain_y)
            h_edge_min = max(0.0, cfg.k_s_min * r_edge)
            required = step_m + h_edge_min
            c_req = required - (g[0]*rx + g[1]*ry)
            c_candidates.append(c_req)
        c = max(c_candidates) if c_candidates else 0.0
        return g, c

    if mode == 'manual' and dir_deg is not None:
        g, c = g_c_from_angle(dir_deg)
        return g, c, dir_deg, c
    elif mode == 'auto':
        cx_room, cy_room = cfg.L/2.0, cfg.W/2.0
        px, py, _ = project_point_to_rect(cx_room, cy_room, sh.rect)
        v = np.array([px - cx_room, py - cy_room], dtype=float)
        if np.linalg.norm(v) < 1e-12:
            vx = (sh.rect[0] + sh.rect[2]/2.0) - cx_room
            vy = (sh.rect[1] + sh.rect[3]/2.0) - cy_room
            v = np.array([vx, vy], dtype=float)
        ang = math.degrees(math.atan2(v[1], v[0]))
        g, c = g_c_from_angle(ang)
        return g, c, ang, c
    else:  # 'opt'
        best = None
        for theta in np.arange(-180.0, 180.0 + 1e-9, cfg.opt_angle_step_deg):
            g, c = g_c_from_angle(theta)
            if (best is None) or (c < best[1] - 1e-12):
                best = (theta, c, g)
        ang, c_best, g_best = best
        return g_best, c_best, ang, c_best

# ------------- Vyöhykkeen testaus -------------

def point_in_zone(x: float, y: float, cfg: Config, zone_is_left: bool) -> bool:
    if not cfg.use_zone_split or cfg.zone_split_line is None:
        # Jos jakoa ei käytetä: määritellään, että vyöhyke A = koko lattia, vyöhyke B = tyhjä
        return True if zone_is_left else False
    x0,y0,x1,y1 = cfg.zone_split_line
    side = signed_side_of_line(x, y, (x0,y0,x1,y1))
    return (side >= 0) if zone_is_left else (side < 0)

# ---------------- Laskenta ----------------

def compute(cfg: Config) -> pd.DataFrame:
    xs = np.arange(0.0, cfg.L + 1e-9, cfg.grid)
    ys = np.arange(0.0, cfg.W + 1e-9, cfg.grid)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    step_m = cfg.step_mm/1000.0

    # Esilasketaan levykaatotasot vyöhykkeille tarvittaessa
    precomp = {}
    for label, zcfg in [('A', cfg.zoneA), ('B', cfg.zoneB)]:
        if zcfg.enabled and (zcfg.shower_idx is not None) and (0 <= zcfg.shower_idx < len(cfg.showers)):
            sh = cfg.showers[zcfg.shower_idx]
            g, c, ang, cval = compute_planar_g_c_for_shower(cfg, sh, zcfg.mode, zcfg.dir_deg)
            precomp[label] = {'g': g, 'c': c, 'angle': ang, 'sh_idx': zcfg.shower_idx, 'enforce_scope': zcfg.enforce_scope}
        else:
            precomp[label] = None

    rows = []
    for j in range(ys.size):
        for i in range(xs.size):
            x = float(X[j, i]); y = float(Y[j, i])
            lowers = []
            uppers = []
            controllers = []

            # Rajaviivojen ohjaus
            sel_s, sel_d = select_targets_by_splitlines(x, y, cfg)

            # Selvitä vyöhyke A/B jäsenyys
            inA = point_in_zone(x, y, cfg, zone_is_left=True)
            inB = point_in_zone(x, y, cfg, zone_is_left=False)

            # 1) Suihkualueet
            for s_idx, sh in enumerate(cfg.showers):
                if inside_rect(x, y, sh.rect):
                    r = math.hypot(x - sh.drain_x, y - sh.drain_y)
                    lv = max(0.0, cfg.k_s_min * r)
                    lowers.append(lv); controllers.append((lv, f'shower#{s_idx}:inside'))
                else:
                    allow_this_shower = (sel_s is None and sel_d is None) or (sel_s == s_idx)

                    # Tarkista, osuuko planar-vyöhyke tähän pisteeseen ja suihkuun
                    applied_planar = False
                    # Zone A
                    if allow_this_shower and inA and precomp['A'] is not None and precomp['A']['sh_idx'] == s_idx:
                        g = precomp['A']['g']; c = precomp['A']['c']
                        lv = g[0]*x + g[1]*y + c
                        lv = max(0.0, lv)
                        lowers.append(lv); controllers.append((lv, f'planarA#S{s_idx}'))
                        applied_planar = True
                        if precomp['A']['enforce_scope']:
                            # ohita muiden suihkujen/drainien ulkoehdot tällä pisteellä
                            pass
                    # Zone B
                    if allow_this_shower and (not applied_planar) and inB and precomp['B'] is not None and precomp['B']['sh_idx'] == s_idx:
                        g = precomp['B']['g']; c = precomp['B']['c']
                        lv = g[0]*x + g[1]*y + c
                        lv = max(0.0, lv)
                        lowers.append(lv); controllers.append((lv, f'planarB#S{s_idx}'))
                        applied_planar = True
                        if precomp['B']['enforce_scope']:
                            pass

                    # Jos planar ei ole aktiivinen tälle pisteelle, käytä vakiologiikkaa
                    if allow_this_shower and (not applied_planar):
                        cx, cy, d_edge = project_point_to_rect(x, y, sh.rect)
                        r_edge = math.hypot(cx - sh.drain_x, cy - sh.drain_y)
                        h_edge_min = max(0.0, cfg.k_s_min * r_edge)
                        lv_base_edge = step_m + h_edge_min
                        lv_slope = step_m + cfg.k_o_min * d_edge
                        lv = max(lv_base_edge, lv_slope)
                        lowers.append(lv); controllers.append((lv, f'shower#{s_idx}:edge+step(cont)'))

            # 2) Muut kaivot: pisteohjaus — vyöhykerajoitus voi ohittaa
            skip_drains = (inA and precomp['A'] is not None and precomp['A']['enforce_scope']) or (inB and precomp['B'] is not None and precomp['B']['enforce_scope'])
            if not skip_drains:
                for d_idx, (dx, dy) in enumerate(cfg.drains):
                    allow_this_drain = (sel_s is None and sel_d is None) or (sel_d == d_idx)
                    if allow_this_drain:
                        d = math.hypot(x - dx, y - dy)
                        lv = cfg.k_o_min * d
                        lowers.append(lv); controllers.append((lv, f'drain#{d_idx}'))

            # 3) Kynnys (AWAY): yläraja FFL <= H - k*d  (puoli + D_max)
            if cfg.threshold is not None:
                x0, y0, x1, y1 = cfg.threshold
                apply_here = True
                if cfg.thr_one_sided:
                    side = signed_side_of_line(x, y, (x0, y0, x1, y1))
                    apply_here = (side >= 0) if cfg.thr_side_is_left else (side < 0)
                if apply_here:
                    d_thr = distance_point_to_segment(x, y, x0, y0, x1, y1)
                    if (cfg.thr_D_max is None) or (d_thr <= cfg.thr_D_max + 1e-12):
                        ub = cfg.thr_H_mm/1000.0 - cfg.thr_k_min * d_thr
                        uppers.append(ub)

            # Lopputasot
            h_lower = max(lowers) if lowers else 0.0
            h_lower = max(0.0, h_lower)
            h_upper = min(uppers) if uppers else float('inf')
            violate = h_lower > h_upper + 1e-9
            ctrl = max(controllers, key=lambda t: t[0])[1] if controllers else 'none'
            h_ffl = h_lower
            h_sfl = h_ffl - cfg.layers_m

            rows.append({
                'x_m': x, 'y_m': y,
                'FFL_m': h_ffl, 'FFL_mm': h_ffl*1000.0,
                'SFL_m': h_sfl, 'SFL_mm': h_sfl*1000.0,
                'controller': ctrl,
                'violates': violate
            })
    return pd.DataFrame(rows)

# ---------------- Visualisointi ----------------

def plot_map(df: pd.DataFrame, cfg: Config, layer='FFL_mm', show_labels=False):
    fig, ax = plt.subplots(figsize=(10,7), dpi=150)
    pivot = df.pivot_table(index='y_m', columns='x_m', values=layer)
    Xs = pivot.columns.values
    Ys = pivot.index.values
    Z = pivot.values
    c = ax.pcolormesh(Xs, Ys, Z, shading='nearest', cmap='viridis')
    cb = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04); cb.set_label(f'{layer} [mm] (0 = suihkukaivo)')
    if show_labels:
        for _, row in df.iterrows():
            ax.text(row['x_m'], row['y_m'], f"{row[layer]:.0f}", color='white', ha='center', va='center', fontsize=7, alpha=0.9)

    # Kehys
    ax.plot([0,cfg.L,cfg.L,0,0],[0,0,cfg.W,cfg.W,0],'k-',lw=1.2)

    # Suihkualueet ja kaivot
    for i, sh in enumerate(cfg.showers):
        x0,y0,w,h = sh.rect
        ax.plot([x0,x0+w,x0+w,x0,x0],[y0,y0,y0+h,y0+h,y0], color='white', lw=2)
        ax.plot(sh.drain_x, sh.drain_y, 'o', color='cyan', markersize=8)
        ax.text(sh.drain_x, sh.drain_y, f'S{i} (0)', color='black', fontsize=8, ha='left', va='bottom')
    if cfg.drains:
        exs=[p[0] for p in cfg.drains]; eys=[p[1] for p in cfg.drains]
        ax.plot(exs, eys, 'o', markerfacecolor='orange', markeredgecolor='k', markersize=7, linestyle='None')
        for i,(dx,dy) in enumerate(cfg.drains):
            ax.text(dx,dy,f'D{i}', color='black', fontsize=8, ha='left', va='bottom')

    # Kynnys
    if cfg.threshold is not None:
        x0,y0,x1,y1 = cfg.threshold
        ax.plot([x0,x1],[y0,y1], color='magenta', lw=3, label=f'Kynnys H={cfg.thr_H_mm:.0f} mm (AWAY)')

    # Vyöhykejakolinja
    if cfg.use_zone_split and cfg.zone_split_line is not None:
        x0,y0,x1,y1 = cfg.zone_split_line
        ax.plot([x0,x1],[y0,y1], color='deepskyblue', lw=2, linestyle='--', label='Levykaato vyöhykeraja')

    # Konfliktit
    viol = df[df['violates']]
    if not viol.empty:
        ax.scatter(viol['x_m'], viol['y_m'], s=28, facecolors='none', edgecolors='red', linewidths=1.2, label='Ristiriita')

    ax.set_aspect('equal', adjustable='box'); ax.set_xlim(0,cfg.L); ax.set_ylim(0,cfg.W)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title('LEVYKAATO + kynnys (puoli & D_max) + vyöhykkeet + kulmaoptimointi')
    ax.grid(True, linestyle=':', alpha=0.35)
    if len(ax.get_legend_handles_labels()[0])>0:
        ax.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', framealpha=0.85)
    fig.tight_layout()
    return fig

# ---------------- Streamlit UI ----------------

def main():
    st.set_page_config(page_title='Lattiakaato PRO2 – FIX2 (levykaato, vyöhykkeet, kynnys)', layout='wide')
    st.title('🛁 Lattiakaatojen suunnittelutyökalu — PRO2 (LEVYKAATO + vyöhykkeet + kynnys)')

    with st.sidebar:
        st.header('Perusmitat')
        L = st.number_input('Lattian pituus L [m] (x)', 0.5, 50.0, 3.0, 0.1)
        W = st.number_input('Lattian leveys W [m] (y)', 0.5, 50.0, 2.4, 0.1)
        grid = st.selectbox('Ruudukon jako', [0.25,0.2,0.5], index=0, format_func=lambda v: f'{int(v*100)} cm')

        st.header('Suihkualueet & kaivo (kaivo = 0‑piste)')
        n_sh = st.number_input('Suihkualueita', 1, 10, 1)
        showers: List[Shower] = []
        for k in range(n_sh):
            st.caption(f'Suihkualue {k+1}')
            c1,c2 = st.columns(2)
            with c1:
                sx = st.number_input(f'x0[{k+1}]', 0.0, L, 0.9, 0.1, key=f'sx{k}')
                sl = st.number_input(f'Pituus[{k+1}]', 0.5, 20.0, 1.2, 0.1, key=f'sl{k}')
            with c2:
                sy = st.number_input(f'y0[{k+1}]', 0.0, W, 0.7, 0.1, key=f'sy{k}')
                sw = st.number_input(f'Leveys[{k+1}]', 0.5, 20.0, 1.0, 0.1, key=f'sw{k}')
            st.write('Kaivon sijainti (oletus keskelle)')
            use_center = st.checkbox(f'Kaivo keskellä [{k+1}]', True, key=f'center{k}')
            if use_center:
                dx = sx + sl/2
                dy = sy + sw/2
            else:
                dx = st.number_input(f'Kaivo x [{k+1}]', 0.0, L, sx + sl/2, 0.01, key=f'dx{k}')
                dy = st.number_input(f'Kaivo y [{k+1}]', 0.0, W, sy + sw/2, 0.01, key=f'dy{k}')
            showers.append(Shower((sx,sy,sl,sw), dx, dy))

        st.header('Muut kaivot (pisteet)')
        n_dr = st.number_input('Muut kaivot', 0, 20, 0)
        drains = []
        for k in range(n_dr):
            c1,c2 = st.columns(2)
            with c1:
                ex = st.number_input(f'Kaivo {k+1} x', 0.0, L, 2.0, 0.1, key=f'ex{k}')
            with c2:
                ey = st.number_input(f'Kaivo {k+1} y', 0.0, W, 2.0, 0.1, key=f'ey{k}')
            drains.append((ex,ey))

        st.header('Kaadot & korkoero')
        sN_min = st.number_input('Suihkualue min-kaato (1:N)', 20.0, 400.0, 50.0, 5.0)
        oN_min = st.number_input('Muu alue min-kaato (1:N)', 50.0, 600.0, 100.0, 5.0)
        step_mm = st.number_input('Korkoero suihku ↔ muu [mm]', 0.0, 200.0, 10.0, 1.0)

        st.header('Kynnys (aina POISPÄIN)')
        use_thr = st.checkbox('Käytä kynnyslinjaa', False)
        thr = None; thr_H = 15.0; thrN = 80.0
        thr_one_sided = False; thr_side_is_left = True; thr_D_max = None
        if use_thr:
            x0 = st.number_input('Kynnys x0', 0.0, L, 0.0, 0.1)
            y0 = st.number_input('Kynnys y0', 0.0, W, 0.0, 0.1)
            x1 = st.number_input('Kynnys x1', 0.0, L, L, 0.1)
            y1 = st.number_input('Kynnys y1', 0.0, W, 0.0, 0.1)
            thr = (x0,y0,x1,y1)
            thr_H = st.number_input('Kynnys FFL [mm] (0‑pisteestä)', -50.0, 500.0, 15.0, 1.0)
            thrN = st.number_input('Min-kaato poispäin (1:N)', 20.0, 1000.0, 80.0, 5.0)
            st.markdown('**Kynnyksen vaikutusalue**')
            thr_one_sided = st.checkbox('Kynnys vaikuttaa vain yhdelle puolelle', False)
            colA, colB = st.columns(2)
            with colA:
                if thr_one_sided:
                    side_label = st.radio('Aktiivinen puoli', ['Vasen puoli', 'Oikea puoli'], index=0)
                    thr_side_is_left = (side_label == 'Vasen puoli')
            with colB:
                use_Dmax = st.checkbox('Rajoita vaikutusmatka D_max', False)
                if use_Dmax:
                    thr_D_max = st.number_input('D_max [m]', 0.1, max(L,W), 0.8, 0.1)

        st.header('Kaatoalueen RAJAVIIVA (ohjaa vasen/oikea)')
        n_sl = st.number_input('Rajaviivoja', 0, 10, 0)
        split_lines: List[SplitLine] = []
        for k in range(n_sl):
            st.caption(f'Rajaviiva {k+1}')
            x0 = st.number_input(f'RL{k+1} x0', 0.0, L, 0.0, 0.1, key=f'rlx0{k}')
            y0 = st.number_input(f'RL{k+1} y0', 0.0, W, W/2, 0.1, key=f'rly0{k}')
            x1 = st.number_input(f'RL{k+1} x1', 0.0, L, L, 0.1, key=f'rlx1{k}')
            y1 = st.number_input(f'RL{k+1} y1', 0.0, W, W/2, 0.1, key=f'rly1{k}')
            colL, colR = st.columns(2)
            with colL:
                ltype = st.selectbox(f'Vasen kohde {k+1}', ['(ei määritelty)','suihku','kaivo'], index=0, key=f'lt{k}')
                lidx = None
                if ltype == 'suihku':
                    lidx = st.number_input(f'Vasen suihkuindeksi {k+1}', 0, max(0,n_sh-1), 0, 1, key=f'ls{k}')
                elif ltype == 'kaivo':
                    lidx = st.number_input(f'Vasen kaivoindeksi {k+1}', 0, max(0,n_dr-1), 0, 1, key=f'ld{k}')
                ltype = None if ltype=='(ei määritelty)' else ('shower' if ltype=='suihku' else 'drain')
            with colR:
                rtype = st.selectbox(f'Oikea kohde {k+1}', ['(ei määritelty)','suihku','kaivo'], index=0, key=f'rt{k}')
                ridx = None
                if rtype == 'suihku':
                    ridx = st.number_input(f'Oikea suihkuindeksi {k+1}', 0, max(0,n_sh-1), 0, 1, key=f'rs{k}')
                elif rtype == 'kaivo':
                    ridx = st.number_input(f'Oikea kaivoindeksi {k+1}', 0, max(0,n_dr-1), 0, 1, key=f'rd{k}')
                rtype = None if rtype=='(ei määritelty)' else ('shower' if rtype=='suihku' else 'drain')
            split_lines.append(SplitLine((x0,y0,x1,y1), ltype, lidx, rtype, ridx))

        st.header('Levykaato — vyöhykkeet A/B')
        use_zone_split = st.checkbox('Käytä levykaaton vyöhykejakoa (A/B)', False)
        zone_split_line = None
        zoneA_is_left = True
        if use_zone_split:
            st.caption('Vyöhykkeen jakolinja (A/B)')
            x0 = st.number_input('Vyöhyke x0', 0.0, L, 0.0, 0.1)
            y0 = st.number_input('Vyöhyke y0', 0.0, W, 0.0, 0.1)
            x1 = st.number_input('Vyöhyke x1', 0.0, L, L, 0.1)
            y1 = st.number_input('Vyöhyke y1', 0.0, W, W/2, 0.1)
            zone_split_line = (x0,y0,x1,y1)
            zoneA_is_left = st.radio('Kumpi puoli on A‑vyöhyke?', ['Vasen', 'Oikea'], index=0) == 'Vasen'

        st.caption('Vyöhyke A — levykaato')
        zA_enabled = st.checkbox('Levykaato A‑vyöhykkeelle', True)
        zA_sh = st.number_input('A: kohdesuihkuindeksi', 0, max(0,n_sh-1), 0, 1)
        zA_mode = st.radio('A: suunta', ['Automaattinen','Manuaalinen','Optimoi'], index=2)
        zA_ang = None
        if zA_mode == 'Manuaalinen':
            zA_ang = st.number_input('A: kulma [°]', -180.0, 180.0, -90.0, 1.0)
        zA_scope = st.checkbox('A: vyöhykerajoitus (ohita muiden suihkujen ja pistekaivojen ehdot A‑vyöhykkeessä)', True)

        st.caption('Vyöhyke B — levykaato')
        zB_enabled = st.checkbox('Levykaato B‑vyöhykkeelle', False)
        zB_sh = st.number_input('B: kohdesuihkuindeksi', 0, max(0,n_sh-1), min(1, max(0,n_sh-1)), 1)
        zB_mode = st.radio('B: suunta', ['Automaattinen','Manuaalinen','Optimoi'], index=0)
        zB_ang = None
        if zB_mode == 'Manuaalinen':
            zB_ang = st.number_input('B: kulma [°]', -180.0, 180.0, 0.0, 1.0)
        zB_scope = st.checkbox('B: vyöhykerajoitus (ohita muiden suihkujen ja pistekaivojen ehdot B‑vyöhykkeessä)', True)

        st.caption('Optimoinnin tarkkuus')
        opt_step = st.slider('Kulma-askeleen koko [°]', 1, 10, 2, 1)

        st.header('Materiaalikerrokset [mm] (FFL → SFL)')
        t_tile = st.number_input('Laatta', 0.0, 30.0, 10.0, 1.0)
        t_adh  = st.number_input('Laattapakki/liima', 0.0, 20.0, 5.0, 1.0)
        t_wpr  = st.number_input('Vedeneristys', 0.0, 10.0, 2.0, 0.5)
        show_labels = st.checkbox('Näytä korkeustekstit', False)
        layer = st.radio('Näytä pinta', ['FFL_mm','SFL_mm'], index=0)

    # Tarkastukset
    for sh in showers:
        x0,y0,w,h = sh.rect
        if x0+w > L or y0+h > W:
            st.error('Suihkualue ulottuu lattian ulkopuolelle.'); st.stop()
        if not (x0 <= sh.drain_x <= x0+w and y0 <= sh.drain_y <= y0+h):
            st.warning('Huom: suihkukaivo on suihkualueen ulkopuolella.')

    # Koosta konfiguraatio
    def map_mode(label:str):
        return {'Automaattinen':'auto','Manuaalinen':'manual','Optimoi':'opt'}[label]

    cfg = Config(
        L=L, W=W, grid=grid,
        showers=showers,
        drains=drains,
        step_mm=step_mm,
        k_s_min=0.0 if sN_min==0 else 1.0/sN_min,
        k_o_min=0.0 if oN_min==0 else 1.0/oN_min,
        threshold=(None if not use_thr else thr),
        thr_H_mm=thr_H,
        thr_k_min=(0.0 if not use_thr else (0.0 if thrN==0 else 1.0/thrN)),
        thr_one_sided=thr_one_sided,
        thr_side_is_left=thr_side_is_left,
        thr_D_max=thr_D_max,
        split_lines=split_lines,
        layers_m=(t_tile + t_adh + t_wpr)/1000.0,
        use_zone_split=use_zone_split,
        zone_split_line=zone_split_line,
        zoneA=PlanarZoneCfg(enabled=zA_enabled, shower_idx=zA_sh, mode=map_mode(zA_mode), dir_deg=zA_ang, zone_is_left=zoneA_is_left, enforce_scope=zA_scope),
        zoneB=PlanarZoneCfg(enabled=zB_enabled, shower_idx=zB_sh, mode=map_mode(zB_mode), dir_deg=zB_ang, zone_is_left=not zoneA_is_left, enforce_scope=zB_scope),
        opt_angle_step_deg=float(opt_step),
    )

    df = compute(cfg)
    fig = plot_map(df, cfg, layer=layer, show_labels=show_labels)
    st.pyplot(fig, clear_figure=True)

    st.subheader('Korkotaulukko')
    st.dataframe(df[['x_m','y_m','FFL_mm','SFL_mm','controller','violates']].sort_values(['y_m','x_m']), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        buf=io.BytesIO(); fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        st.download_button('⤓ PNG', data=buf.getvalue(), file_name='lattiakaadot_levykaato_vyohykkeet_thr.png', mime='image/png')
    with c2:
        st.download_button('⤓ CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='korkopisteet_levykaato_vyohykkeet_thr.csv', mime='text/csv')


if __name__ == '__main__':
    main()
