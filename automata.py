"""
Autòmat Cel·lular de Wolfram — amb Gra Guixut K=2
===================================================
La finestra mostra tres panells en temps real:
  1. CA original
  2. CA renormalitzat per gra guixut K=2 (vot de majoria per blocs)
  3. Diferències entre el CA original mostrejat i el renormalitzat
     + regla de Wolfram inferida i precisió de l'ajust
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import matplotlib.patches as mpatches


# ═══════════════════════════════════════════════════════════════
#  Lògica del CA
# ═══════════════════════════════════════════════════════════════

def make_rule_table(num: int) -> dict:
    pats = ['111', '110', '101', '100', '011', '010', '001', '000']
    return {p: (num >> (7 - i)) & 1 for i, p in enumerate(pats)}


def run_ca(rules_list, n_cel=150, n_gen=80,
           frontera='periodic', inici='single', seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if inici == 'single':
        row = np.zeros(n_cel, dtype=np.int8)
        row[n_cel // 2] = 1
    elif inici == 'random':
        row = rng.integers(0, 2, size=n_cel, dtype=np.int8)
    else:
        row = np.array([i % 2 for i in range(n_cel)], dtype=np.int8)

    tables = [make_rule_table(r) for r in rules_list]
    franja = n_cel // len(rules_list)
    assignacio = np.zeros(n_cel, dtype=int)
    for k in range(len(rules_list)):
        fi = (k + 1) * franja if k < len(rules_list) - 1 else n_cel
        assignacio[k * franja:fi] = k

    grid = np.zeros((n_gen, n_cel), dtype=np.int8)
    grid[0] = row

    for g in range(1, n_gen):
        prev = grid[g - 1]
        nova = np.zeros(n_cel, dtype=np.int8)
        for i in range(n_cel):
            c = prev[i]
            if frontera == 'periodic':
                l, r = prev[(i - 1) % n_cel], prev[(i + 1) % n_cel]
            elif frontera == 'mirror':
                l = prev[1] if i == 0 else prev[i - 1]
                r = prev[-2] if i == n_cel - 1 else prev[i + 1]
            else:
                l = 0 if i == 0 else prev[i - 1]
                r = 0 if i == n_cel - 1 else prev[i + 1]
            nova[i] = tables[assignacio[i]][f"{l}{c}{r}"]
        grid[g] = nova

    return grid


def gra_guixut(grid: np.ndarray, K: int = 2):
    """
    Renormalitza el grid agrupant K cel·les per bloc (vot de majoria).
    Retorna:
        cg          — grid renormalitzat  (n_gen x n_blocs)
        regla_inf   — número de regla Wolfram inferida (0-255)
        precisio    — % de transicions que concorden amb la regla inferida
        comptes     — dict detallat de vots per patró
    """
    n_gen, n_cel = grid.shape
    n_blocs = n_cel // K

    # Renormalització per majoria
    cg = np.zeros((n_gen, n_blocs), dtype=np.int8)
    for g in range(n_gen):
        blocs = grid[g, :n_blocs * K].reshape(n_blocs, K)
        cg[g] = (blocs.sum(axis=1) > K / 2).astype(np.int8)

    # Inferència de la regla
    pats = ['111', '110', '101', '100', '011', '010', '001', '000']
    comptes = {p: {0: 0, 1: 0} for p in pats}

    for g in range(n_gen - 1):
        for i in range(n_blocs):
            l = cg[g, (i - 1) % n_blocs]
            c = cg[g, i]
            r = cg[g, (i + 1) % n_blocs]
            comptes[f"{l}{c}{r}"][int(cg[g + 1, i])] += 1

    inferida = {
        p: (1 if comptes[p][1] >= comptes[p][0] else 0)
        for p in pats
    }
    regla_inf = sum(inferida[p] << (7 - i) for i, p in enumerate(pats))

    total   = sum(comptes[p][0] + comptes[p][1] for p in pats)
    encerts = sum(comptes[p][inferida[p]] for p in pats)
    precisio = encerts / total * 100 if total else 0.0

    return cg, regla_inf, precisio, comptes


def rule_description(num: int) -> str:
    known = {
        0: "mort total", 1: "escacat", 18: "triangles", 22: "fractals",
        30: "caòtica", 45: "caòtica", 54: "complex",
        60: "XOR/fractal", 90: "Sierpinski", 110: "universal",
        126: "complex", 150: "fractal XOR", 184: "tràfic", 254: "tot ple",
    }
    return known.get(num, "")


# ═══════════════════════════════════════════════════════════════
#  Interfície gràfica
# ═══════════════════════════════════════════════════════════════

class AutomatApp:
    COLORS_ZONE = ['#2563eb', '#dc2626', '#16a34a', '#9333ea']

    def __init__(self):
        self.rules_list = [30]
        self.n_cel  = 150
        self.n_gen  = 80
        self.K      = 2
        self.frontera = 'periodic'
        self.inici    = 'single'
        self.seed     = 42
        self.grid     = None

        self._build_ui()
        self._run_and_draw()
        plt.show()

    # -----------------------------------------------------------
    #  Construcció de la finestra
    # -----------------------------------------------------------

    def _build_ui(self):
        self.fig = plt.figure(figsize=(18, 8), facecolor='#fafaf8')
        self.fig.canvas.manager.set_window_title(
            'Autòmat Cel·lular de Wolfram — Gra Guixut K=2')

        # Layout: controls | CA original | GG | diff
        gs = gridspec.GridSpec(
            1, 4, width_ratios=[0.85, 2, 2, 2],
            left=0.02, right=0.99, top=0.95, bottom=0.15, wspace=0.12)

        gs_ctrl = gridspec.GridSpecFromSubplotSpec(
            13, 1, subplot_spec=gs[0], hspace=0.5)

        self.ax_ca   = self.fig.add_subplot(gs[1])
        self.ax_cg   = self.fig.add_subplot(gs[2])
        self.ax_diff = self.fig.add_subplot(gs[3])
        for ax in (self.ax_ca, self.ax_cg, self.ax_diff):
            ax.set_facecolor('#f8f7f3')

        # Títol controls
        ax_t = self.fig.add_subplot(gs_ctrl[0])
        ax_t.axis('off')
        ax_t.text(0.5, 0.5, 'Controls', ha='center', va='center',
                  fontsize=10, fontweight='bold', color='#1a1a18')

        # Sliders
        sliders_cfg = [
            ('Cel·les',     1, 2,  'n_cel', 30,  300, 150, 10,  '#2563eb'),
            ('Generacions', 3, 4,  'n_gen', 20,  200,  80, 10,  '#2563eb'),
            ('Bloc K',      5, 6,  'K',      2,    8,   2,  1,  '#9333ea'),
            ('Seme aleat.', 7, 8,  'seed',   0,   99,  42,  1,  '#888780'),
        ]
        self._sliders = {}
        for label, r_lbl, r_sl, attr, vmin, vmax, vinit, vstep, col \
                in sliders_cfg:
            ax_lbl = self.fig.add_subplot(gs_ctrl[r_lbl])
            ax_lbl.axis('off')
            ax_lbl.text(0.05, 0.5, label, va='center',
                        fontsize=8, color='#5f5e5a')
            ax_sl = self.fig.add_axes(self._crect(gs_ctrl[r_sl]))
            sl = Slider(ax_sl, '', vmin, vmax, valinit=vinit,
                        valstep=vstep, color=col)
            sl.valtext.set_text(str(vinit))
            sl.on_changed(lambda v, s=sl, a=attr:
                          self._on_slider(v, s, a))
            self._sliders[attr] = sl

        # Frontera radio
        ax_r1 = self.fig.add_subplot(gs_ctrl[9])
        ax_r1.axis('off')
        ax_r1.text(0.05, 0.5, 'Frontera', va='center',
                   fontsize=8, color='#5f5e5a')
        ax_radio1 = self.fig.add_axes(self._crect(gs_ctrl[10], h=0.085))
        self.radio_frontera = RadioButtons(
            ax_radio1, ('periodic', 'zero', 'mirror'), activecolor='#2563eb')
        for lbl in self.radio_frontera.labels:
            lbl.set_fontsize(7)
        self.radio_frontera.on_clicked(self._on_frontera)

        # Inici radio
        ax_r2 = self.fig.add_subplot(gs_ctrl[11])
        ax_r2.axis('off')
        ax_r2.text(0.05, 0.5, 'Inici', va='center',
                   fontsize=8, color='#5f5e5a')
        ax_radio2 = self.fig.add_axes(self._crect(gs_ctrl[12], h=0.085))
        self.radio_inici = RadioButtons(
            ax_radio2, ('single', 'random', 'alternating'),
            activecolor='#2563eb')
        for lbl in self.radio_inici.labels:
            lbl.set_fontsize(7)
        self.radio_inici.on_clicked(self._on_inici)

        # TextBox + botons
        self.ax_tb = self.fig.add_axes([0.015, 0.07, 0.10, 0.04])
        self.tb_rule = TextBox(self.ax_tb, '', initial='110',
                               textalignment='center')
        self.tb_rule.on_submit(self._on_add_rule)

        self.ax_btn_add = self.fig.add_axes([0.125, 0.07, 0.065, 0.04])
        self.btn_add = Button(self.ax_btn_add, '+ Regla',
                              color='#e8e6df', hovercolor='#d3d1c7')
        self.btn_add.label.set_fontsize(7)
        self.btn_add.on_clicked(
            lambda e: self._on_add_rule(self.tb_rule.text))

        self.ax_btn_clr = self.fig.add_axes([0.015, 0.02, 0.085, 0.04])
        self.btn_clr = Button(self.ax_btn_clr, 'Netejar regles',
                              color='#faeeda', hovercolor='#fac775')
        self.btn_clr.label.set_fontsize(7)
        self.btn_clr.on_clicked(self._on_clear_rules)

        self.ax_btn_save = self.fig.add_axes([0.115, 0.02, 0.075, 0.04])
        self.btn_save = Button(self.ax_btn_save, 'Desa PNG',
                               color='#eaf3de', hovercolor='#c0dd97')
        self.btn_save.label.set_fontsize(7)
        self.btn_save.on_clicked(self._on_save)

        # Etiqueta regles actives
        self.ax_rlbl = self.fig.add_axes([0.015, 0.115, 0.2, 0.03])
        self.ax_rlbl.axis('off')
        self.lbl_rules = self.ax_rlbl.text(
            0.0, 0.5, '', va='center', fontsize=7, color='#1a1a18')

        # Taula de vots (baix a la dreta)
        self.ax_info = self.fig.add_axes([0.55, 0.005, 0.44, 0.13])
        self.ax_info.axis('off')
        self.lbl_info = self.ax_info.text(
            0.0, 1.0, '', va='top', fontsize=7,
            color='#1a1a18', family='monospace')

    def _crect(self, spec, h=0.022):
        r = spec.get_position(self.fig)
        return [r.x0 + 0.005, r.y0, r.width - 0.01, h]

    # -----------------------------------------------------------
    #  Callbacks
    # -----------------------------------------------------------

    def _on_slider(self, val, slider, attr):
        v = int(val)
        slider.valtext.set_text(str(v))
        setattr(self, attr, v)
        self._run_and_draw()

    def _on_frontera(self, label):
        self.frontera = label
        self._run_and_draw()

    def _on_inici(self, label):
        self.inici = label
        self._run_and_draw()

    def _on_add_rule(self, text):
        try:
            r = int(text.strip())
            assert 0 <= r <= 255
        except (ValueError, AssertionError):
            return
        if r not in self.rules_list:
            self.rules_list.append(r)
        self._run_and_draw()

    def _on_clear_rules(self, event):
        self.rules_list = [self.rules_list[0]]
        self._run_and_draw()

    def _on_save(self, event):
        fname = (f"CA_{'_'.join(map(str, self.rules_list))}"
                 f"_{self.frontera}_{self.inici}_K{self.K}.png")
        self.fig.savefig(fname, dpi=150, bbox_inches='tight',
                         facecolor=self.fig.get_facecolor())
        print(f"Desat: {fname}")

    # -----------------------------------------------------------
    #  Dibuix
    # -----------------------------------------------------------

    def _run_and_draw(self):
        self.grid = run_ca(
            self.rules_list, self.n_cel, self.n_gen,
            self.frontera, self.inici, self.seed)

        K = self.K
        n_cel_cg = (self.n_cel // K) * K
        cg, regla_inf, precisio, comptes = gra_guixut(
            self.grid[:, :n_cel_cg], K)

        # ── Panell 1: CA original ──────────────────────────
        self.ax_ca.cla()
        self.ax_ca.set_facecolor('#f8f7f3')

        if len(self.rules_list) == 1:
            self.ax_ca.imshow(self.grid, cmap='binary',
                              interpolation='nearest', aspect='auto')
            desc = rule_description(self.rules_list[0])
            title1 = f"Regla {self.rules_list[0]}"
            if desc:
                title1 += f"  ({desc})"
        else:
            rgb = np.ones((*self.grid.shape, 3))
            franja = self.n_cel // len(self.rules_list)
            for k, col_hex in enumerate(
                    self.COLORS_ZONE[:len(self.rules_list)]):
                fi = ((k + 1) * franja if k < len(self.rules_list) - 1
                      else self.n_cel)
                rv, gv, bv = _hex_to_rgb(col_hex)
                mask = self.grid[:, k * franja:fi] == 1
                for ch, v in enumerate([rv, gv, bv]):
                    rgb[:, k * franja:fi, ch] = np.where(mask, v, 0.973)
                if k > 0:
                    self.ax_ca.axvline(k * franja - 0.5,
                                       color='#888780', lw=0.8,
                                       ls='--', alpha=0.6)
            self.ax_ca.imshow(rgb, interpolation='nearest', aspect='auto')
            patches = [mpatches.Patch(color=self.COLORS_ZONE[i],
                                      label=f'Regla {r}')
                       for i, r in enumerate(self.rules_list)]
            self.ax_ca.legend(handles=patches, loc='upper right',
                              fontsize=7, framealpha=0.85)
            title1 = "Combinació: " + ' + '.join(map(str, self.rules_list))

        self._style_ax(self.ax_ca, title1, 'Cel·la')

        # ── Panell 2: Gra guixut ───────────────────────────
        self.ax_cg.cla()
        self.ax_cg.set_facecolor('#f8f7f3')
        self.ax_cg.imshow(cg, cmap='binary',
                          interpolation='nearest', aspect='auto')
        desc_inf = rule_description(regla_inf)
        title2 = (f"Gra guixut K={K}  →  regla inferida {regla_inf}"
                  + (f"  ({desc_inf})" if desc_inf else ""))
        self._style_ax(self.ax_cg,
                       f"{title2}\nPrecisió ajust: {precisio:.1f}%",
                       f'Bloc (K={K})')

        # ── Panell 3: Diferències ──────────────────────────
        self.ax_diff.cla()
        self.ax_diff.set_facecolor('#f8f7f3')

        ca_down = self.grid[:, :n_cel_cg:K][:, :cg.shape[1]]
        diff = np.abs(ca_down.astype(int) - cg.astype(int))
        n_diff   = int(diff.sum())
        pct_diff = n_diff / diff.size * 100

        self.ax_diff.imshow(diff, cmap='Reds',
                            interpolation='nearest',
                            aspect='auto', vmin=0, vmax=1)
        self._style_ax(self.ax_diff,
                       f"Diferències CA↓ vs GG\n"
                       f"{n_diff} cel·les ({pct_diff:.1f}%) discordants",
                       f'Bloc (K={K})')

        # ── Taula de vots ──────────────────────────────────
        pats = ['111', '110', '101', '100', '011', '010', '001', '000']
        lines = [f"{'Patró':>5}  {'→0':>5}  {'→1':>5}  {'inf':>4}",
                 "─" * 28]
        for p in pats:
            v0  = comptes[p][0]
            v1  = comptes[p][1]
            inf = '1' if v1 >= v0 else '0'
            lines.append(f"  {p}   {v0:5d}  {v1:5d}    {inf}")
        self.lbl_info.set_text('\n'.join(lines))

        # Etiqueta regles actives
        self.lbl_rules.set_text(
            'Regles: ' + ', '.join(map(str, self.rules_list)))

        self.fig.canvas.draw_idle()

    def _style_ax(self, ax, title, xlabel):
        ax.set_title(title, fontsize=8, color='#2c2c2a', pad=5)
        ax.set_xlabel(xlabel, fontsize=8, color='#5f5e5a')
        ax.set_ylabel('Generació', fontsize=8, color='#5f5e5a')
        ax.tick_params(labelsize=7, colors='#888780')
        for sp in ax.spines.values():
            sp.set_edgecolor('#d3d1c7')


# ═══════════════════════════════════════════════════════════════
#  Utilitats
# ═══════════════════════════════════════════════════════════════

def _hex_to_rgb(h: str):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# ═══════════════════════════════════════════════════════════════
#  Punt d'entrada
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    AutomatApp()