# -*- coding: utf-8 -*-
"""
Lattiakaato PRO2 – FIX2 — Threshold scope & debug overlay

Laajennukset verrattuna vakaaseen versioon:
1) Kynnys vain toiselle puolelle (yksipuolinen vaikutus): vasen/oikea puoli kynnyssuunnasta katsottuna.
2) Kynnyksen enimmäisetäisyys D_max [m]: kynnys rajoittaa vain tähän asti, kauempana ei ylärajaa.
3) Debug-visualisointi: korosta alueet, joissa kynnys käytännössä ohjaa (yläraja ~ alaraja, marginaali ≤ asetettu mm).

Muutoin logiikka on identtinen vakaaseen "corrected_final_logic" -versioon.
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
from dataclasses import dataclass
from typing import Optional, List, Tuple

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
    # Uudet asetukset
    thr_one_sided: bool                # True = vaikuttaa vain yhdellä puolella
    thr_side_is_left: bool             # True = vasen puoli aktiivinen, False = oikea
    thr_D_max: Optional[float]         # m, None = rajaton
    # Kaatoalueiden rajaviivat
    split_lines: List[SplitLine]
    # Kerrokset (valukorko laskentaa varten tarvittaessa)
    layers_m: float

# ------------- Rajaviivaohjaus -------------

def select_targets_by_splitlines(x, y, cfg: Config):
    """Palauta (selected_shower_idx, selected_drain_idx) ohjauksen perusteella.
    Vain toinen voi olla valittu puolelle (tai None jos ei määritelty).
    Jos useita viivoja päällekkäin, viimeisin (UI-järjestyksessä) voittaa.
    """
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

# ---------------- Laskenta ----------------

def compute(cfg: Config) -> pd.DataFrame:
    xs = np.arange(0.0, cfg.L + 1e-9, cfg.grid)
    ys = np.arange(0.0, cfg.W + 1e-9, cfg.grid)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    step_m = cfg.step_mm / 1000.0
    rows = []

    for j in range(ys.size):
        for i in range(xs.size):
            x = float(X[j, i]); y = float(Y[j, i])
            lowers = []   # alarajat (>=)
            uppers = []   # ylärajat (<=)
            controllers = []

            # Rajaviivojen ohjaus (jos määritelty tälle pisteelle)
            sel_s, sel_d = select_targets_by_splitlines(x, y, cfg)

            # 1) Suihkualueet
            for s_idx, sh in enumerate(cfg.showers):
                if inside_rect(x, y, sh.rect):
                    r = math.hypot(x - sh.drain_x, y - sh.drain_y)
                    lv = cfg.k_s_min * r
                    lv = max(0.0, lv)
                    lowers.append(lv); controllers.append((lv, f'shower#{s_idx}:inside'))
                else:
                    allow_this_shower = (sel_s is None and sel_d is None) or (sel_s == s_idx)
                    if allow_this_shower:
                        d_edge = distance_point_to_rect(x, y, sh.rect)
                        lv = cfg.k_o_min * d_edge if step_m == 0.0 else step_m + cfg.k_o_min * d_edge
                        lowers.append(lv); controllers.append((lv, f'shower#{s_idx}:edge+step'))

            # 2) Muut kaivot: pisteohjaus, ilman step-lisää
            for d_idx, (dx, dy) in enumerate(cfg.drains):
                allow_this_drain = (sel_s is None and sel_d is None) or (sel_d == d_idx)
                if allow_this_drain:
                    d = math.hypot(x - dx, y - dy)
                    lv = cfg.k_o_min * d
                    lowers.append(lv); controllers.append((lv, f'drain#{d_idx}'))

            # 3) Kynnys (AWAY): yläraja FFL <= H - k*d, laajennettu
            upper_applied = False
            if cfg.threshold is not None:
                x0, y0, x1, y1 = cfg.threshold
                # Tarkista yksipuolisuus
                if cfg.thr_one_sided:
                    side = signed_side_of_line(x, y, (x0, y0, x1, y1))
                    apply_here = (side >= 0) if cfg.thr_side_is_left else (side < 0)
                else:
                    apply_here = True
                if apply_here:
                    d_thr = distance_point_to_segment(x, y, x0, y0, x1, y1)
                    if (cfg.thr_D_max is None) or (d_thr <= cfg.thr_D_max + 1e-12):
                        ub = cfg.thr_H_mm/1000.0 - cfg.thr_k_min * d_thr
                        uppers.append(ub)
                        upper_applied = True

            # Lopputulos
            h_lower = max(lowers) if lowers else 0.0
            h_lower = max(0.0, h_lower)
            h_upper = min(uppers) if uppers else float('inf')
            violate = h_lower > h_upper + 1e-9
            ctrl = max(controllers, key=lambda t: t[0])[1] if controllers else 'none'
            h_ffl = h_lower
            h_sfl = h_ffl - cfg.layers_m

            # Debug: kuinka lähellä kynnys rajaa (jos sovellettu)
            upper_margin_mm = float('inf')
            if upper_applied and not math.isinf(h_upper):
                upper_margin_mm = (h_upper - h_lower) * 1000.0

            rows.append({
                'x_m': x, 'y_m': y,
                'FFL_m': h_ffl, 'FFL_mm': h_ffl*1000.0,
                'SFL_m': h_sfl, 'SFL_mm': h_sfl*1000.0,
                'controller': ctrl,
                'violates': violate,
                'upper_margin_mm': upper_margin_mm
            })
    return pd.DataFrame(rows)

# ---------------- Visualisointi ----------------

def plot_map(df: pd.DataFrame, cfg: Config, layer='FFL_mm', show_labels=False,
             show_thr_control=False, thr_margin_mm=2.0):
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

    # Kynnyslinja
    if cfg.threshold is not None:
        x0,y0,x1,y1 = cfg.threshold
        ax.plot([x0,x1],[y0,y1], color='magenta', lw=3, label=f'Kynnys H={cfg.thr_H_mm:.0f} mm (AWAY)')
        # piirrä muutama nuoli poispäin kynnyksestä
        nx, ny = (y0 - y1), (x1 - x0)
        norm = math.hypot(nx, ny) or 1.0
        nx, ny = nx/norm, ny/norm
        cx, cy = (x0+x1)/2, (y0+y1)/2
        for k in range(-2,3):
            tx = cx + (x1 - x0)*(k/10.0)
            ty = cy + (y1 - y0)*(k/10.0)
            ax.arrow(tx, ty, 0.15*nx, 0.15*ny, head_width=0.05, head_length=0.05, fc='magenta', ec='magenta')

    # Konfliktit
    viol = df[df['violates']]
    if not viol.empty:
        ax.scatter(viol['x_m'], viol['y_m'], s=28, facecolors='none', edgecolors='red', linewidths=1.2, label='Ristiriita')

    # Debug: korosta missä kynnys ohjaa (ylärajan marginaali pieni)
    if show_thr_control and cfg.threshold is not None:
        mask = df['upper_margin_mm'].apply(lambda v: (v != float('inf')) and (v >= 0) and (v <= thr_margin_mm))
        near = df[mask]
        if not near.empty:
            ax.scatter(near['x_m'], near['y_m'], s=22, c='magenta', alpha=0.35, marker='s', label=f'Kynnys ohjaa (≤{thr_margin_mm:.0f} mm)')

    ax.set_aspect('equal', adjustable='box'); ax.set_xlim(0,cfg.L); ax.set_ylim(0,cfg.W)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title('Lattiakaatojen korkomalli — Threshold scope')
    ax.grid(True, linestyle=':', alpha=0.35)
    if len(ax.get_legend_handles_labels()[0])>0:
        ax.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', framealpha=0.85)
    fig.tight_layout()
    return fig

# ---------------- Streamlit UI ----------------

def main():
    st.set_page_config(page_title='Lattiakaato PRO2 – FIX2 (threshold scope)', layout='wide')
    st.title('🛁 Lattiakaatojen suunnittelutyökalu — PRO2 (kynnys: puoli, etäisyys, overlay)')

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
            st.write('Kaivon sijainti (oletus keskelle):')
            use_center = st.checkbox(f'Kaivo keskellä [{k+1}]', True, key=f'center{k}')
            if use_center:
                dx = sx + sl/2; dy = sy + sw/2
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
            st.markdown('**Vaikutusalue**')
            thr_one_sided = st.checkbox('Kynnys vaikuttaa vain yhdelle puolelle', False)
            colA, colB = st.columns(2)
            with colA:
                if thr_one_sided:
                    side_label = st.radio('Aktiivinen puoli', ['Vasen puoli', 'Oikea puoli'], index=0,
                        help='Määrittyy kynnyksen suuntavektorin (x0,y0)→(x1,y1) mukaan. Vasen = positiivinen puoli.')
                    thr_side_is_left = (side_label == 'Vasen puoli')
            with colB:
                use_Dmax = st.checkbox('Rajoita vaikutusmatka D_max', False)
                if use_Dmax:
                    thr_D_max = st.number_input('D_max [m]', 0.1, max(L,W), 0.8, 0.1)

        st.header('Visualisointi')
        show_labels = st.checkbox('Näytä korkeustekstit', False)
        show_thr_control = st.checkbox('Korosta alue, jossa kynnys ohjaa', True)
        thr_margin_mm = st.slider('Kynnyksen ohjausalue: marginaali [mm]', 1, 10, 2, 1)
        layer = st.radio('Näytä pinta', ['FFL_mm','SFL_mm'], index=0)

    # Tarkastukset
    for sh in showers:
        x0,y0,w,h = sh.rect
        if x0+w > L or y0+h > W:
            st.error('Suihkualue ulottuu lattian ulkopuolelle.'); st.stop()
        if not (x0 <= sh.drain_x <= x0+w and y0 <= sh.drain_y <= y0+h):
            st.warning('Huom: suihkukaivo on suihkualueen ulkopuolella.')

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
        split_lines=[],
        layers_m=0.0,
    )

    df = compute(cfg)
    fig = plot_map(df, cfg, layer=layer, show_labels=show_labels,
                   show_thr_control=show_thr_control, thr_margin_mm=thr_margin_mm)
    st.pyplot(fig, clear_figure=True)

    st.subheader('Korkotaulukko')
    st.dataframe(df[['x_m','y_m','FFL_mm','SFL_mm','upper_margin_mm','controller','violates']].sort_values(['y_m','x_m']), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        buf=io.BytesIO(); fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        st.download_button('⤓ PNG', data=buf.getvalue(), file_name='lattiakaadot_threshold_scope.png', mime='image/png')
    with c2:
        st.download_button('⤓ CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='korkopisteet_threshold_scope.csv', mime='text/csv')


if __name__ == '__main__':
    main()
