"""
Simulador de propagació d'incendi forestal — model m:n-CA^k
Llegeix fitxers en format IDRISI32 (.doc/.img) i vectorial IDRISI31 (.dvc/.vec)

Fitxers d'entrada:
  - Initialize.doc / Initialize.img   → capa inicial (tipus de terreny / humitat)
  - vegetation.dvc / vegetation.vec   → capa de vegetació (polígons vectorials)

Ús:
  python wildfire_ca.py
  python wildfire_ca.py --steps 30 --ignite 5 5
  python wildfire_ca.py --help
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation


# ─────────────────────────────────────────────
#  CONSTANTS DE L'AUTÒMAT CEL·LULAR
# ─────────────────────────────────────────────
class CellState:
    EMPTY   = 0   # pendent de cremar
    MOIST   = 1   # escalfant-se (eliminant humitat)
    BURNING = 2   # cremant activament → propaga foc
    BURNED  = 3   # ja cremat (cendra)


# Mapa de codis de terreny de Initialize.img → humitat base (hores)
TERRAIN_HUMIDITY = {
    "LL": 0,   # llis / sense vegetació
    "CT": 1,
    "TE": 1,
    "GR": 2,   # gespa / herba
    "CC": 2,
    "AA": 3,
    "BS": 3,   # bosc sec
    "BN": 4,   # bosc normal
    "BC": 5,   # bosc costaner (més humit)
    "CAT": 2,  # Catalunya genèric
}

# Mapa de polígon ID (vegetation.vec) → hores de combustió
POLYGON_VEGETATION = {
    1: 10,   # polígon 1 - vegetació densa (10 hores)
    2: 5,    # polígon 2 - vegetació mitjana (5 hores)
    20: 3,   # polígon 20 - vegetació baixa (3 hores)
}
DEFAULT_VEGETATION = 4  # valor per defecte si cap polígon cobreix la cel·la


# ─────────────────────────────────────────────
#  LECTURA DE FITXERS IDRISI32
# ─────────────────────────────────────────────
def parse_idrisi_doc(path):
    """Llegeix un fitxer de capçalera IDRISI32 (.doc) i retorna un dict."""
    meta = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip().lower()] = val.strip()
    return meta


def parse_idrisi_img_ascii(path):
    """
    Llegeix un fitxer de dades IDRISI32 ascii (.img).
    Cada línia és un valor (string o numèric).
    Retorna una llista de valors.
    """
    values = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            v = line.strip()
            if v:
                values.append(v)
    return values


def build_humidity_grid(meta, img_values, rows, cols):
    """
    Construeix la quadrícula de humitat (rows×cols) a partir
    dels codis de terreny del fitxer .img.
    Si hi ha menys valors que cel·les, es repeteix cíclicament.
    """
    grid = np.zeros((rows, cols), dtype=int)
    n = len(img_values)
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) % n
            code = img_values[idx].upper()
            grid[r, c] = TERRAIN_HUMIDITY.get(code, 1)
    return grid


# ─────────────────────────────────────────────
#  LECTURA DE FITXERS IDRISI31 VECTORIALS
# ─────────────────────────────────────────────
def parse_vec(path):
    """
    Llegeix un fitxer .vec de format IDRISI31.
    Format per a cada polígon:
        <id> <num_punts>
        x0 y0
        x1 y1
        ...
        0 0    ← separador final
    Retorna llista de dicts: [{id, points: [(x,y), ...]}, ...]
    """
    polygons = []
    with open(path, "r", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 2:
            i += 1
            continue
        try:
            poly_id = int(parts[0])
            num_pts = int(parts[1])
        except ValueError:
            i += 1
            continue

        i += 1
        points = []
        while i < len(lines):
            xy = lines[i].split()
            i += 1
            if len(xy) < 2:
                continue
            x, y = float(xy[0]), float(xy[1])
            if x == 0.0 and y == 0.0:
                break
            points.append((x, y))

        if points:
            polygons.append({"id": poly_id, "points": points})

    return polygons


def point_in_polygon(px, py, polygon_points):
    """
    Ray-casting per comprovar si el punt (px, py) és dins del polígon.
    """
    n = len(polygon_points)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def build_vegetation_grid(polygons, rows, cols, min_x, max_x, min_y, max_y):
    """
    Rasteritza els polígons vectorials a una quadrícula rows×cols.
    Cada cel·la rep el valor de vegetació del polígon que la cobreix.
    """
    grid = np.full((rows, cols), DEFAULT_VEGETATION, dtype=int)

    # Normalitzem les coordenades del vec (0–100) a la quadrícula
    for r in range(rows):
        for c in range(cols):
            # Centre de la cel·la en coordenades del vec (0–100)
            px = (c + 0.5) / cols * (max_x - min_x) + min_x
            py = (r + 0.5) / rows * (max_y - min_y) + min_y

            for poly in polygons:
                if point_in_polygon(px, py, poly["points"]):
                    grid[r, c] = POLYGON_VEGETATION.get(poly["id"], DEFAULT_VEGETATION)
                    break  # el primer polígon que coincideix guanya

    return grid


# ─────────────────────────────────────────────
#  AUTÒMAT CEL·LULAR — REGLES DE TRANSICIÓ
# ─────────────────────────────────────────────
class WildfireCA:
    """
    Model m:n-CA^k per a la propagació d'un incendi forestal.

    Capes:
      - state      : CellState (EMPTY / MOIST / BURNING / BURNED)
      - hum        : humitat restant (hores)
      - veg        : vegetació restant (hores de combustió)
      - hum_init   : humitat inicial (per a reset)
      - veg_init   : vegetació inicial (per a reset)
      - burn_timer : comptador de passos cremant

    Transicions:
      EMPTY → MOIST    si almenys un veí crema i hum > 0
      EMPTY → BURNING  si almenys un veí crema i hum == 0
      MOIST → BURNING  quan hum_timer >= hum_init
      BURNING → BURNED quan burn_timer >= veg_init
    """

    def __init__(self, humidity_grid, vegetation_grid, wind_dir=(0, 0)):
        self.rows, self.cols = humidity_grid.shape
        self.hum_init  = humidity_grid.copy()
        self.veg_init  = vegetation_grid.copy()
        self.wind_dir  = wind_dir   # (row_delta, col_delta) favorit
        self.reset()

    def reset(self):
        self.state      = np.full((self.rows, self.cols), CellState.EMPTY, dtype=int)
        self.hum        = self.hum_init.copy().astype(float)
        self.veg        = self.veg_init.copy().astype(float)
        self.burn_timer = np.zeros((self.rows, self.cols), dtype=float)
        self.hum_timer  = np.zeros((self.rows, self.cols), dtype=float)
        self.step       = 0
        self.history    = []

    def ignite(self, row, col):
        """Encén manualment la cel·la (row, col)."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.state[row, col]      = CellState.BURNING
            self.burn_timer[row, col] = 0
            print(f"  Foc iniciat a [{row},{col}] — "
                  f"vegetació:{self.veg_init[row,col]}h, "
                  f"humitat:{self.hum_init[row,col]}h")

    def _has_burning_neighbor(self, r, c):
        """Comprova si alguna cel·la adjacent (8-veïns) crema."""
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.state[nr, nc] == CellState.BURNING:
                        return True
        return False

    def _spread_prob(self, r, c, dr, dc):
        """
        Probabilitat base de propagació, modificada pel vent.
        wind_dir = (fila_delta, col_delta): direcció preferent.
        """
        base = 0.85
        wr, wc = self.wind_dir
        if wr != 0 or wc != 0:
            # si la propagació va en la mateixa direcció que el vent → +boost
            dot = dr * wr + dc * wc
            norm = (wr**2 + wc**2) ** 0.5
            if norm > 0:
                boost = dot / norm * 0.12
                base = min(0.98, max(0.3, base + boost))
        return base

    def advance(self):
        """Executa un pas de l'autòmat cel·lular."""
        new_state      = self.state.copy()
        new_hum        = self.hum.copy()
        new_veg        = self.veg.copy()
        new_burn_timer = self.burn_timer.copy()
        new_hum_timer  = self.hum_timer.copy()

        for r in range(self.rows):
            for c in range(self.cols):
                s = self.state[r, c]

                # ── Cel·la cremant ──────────────────────────
                if s == CellState.BURNING:
                    new_burn_timer[r, c] += 1
                    if new_burn_timer[r, c] >= self.veg_init[r, c]:
                        new_state[r, c] = CellState.BURNED
                    else:
                        # propaga a veïns
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                                    continue
                                if self.state[nr, nc] != CellState.EMPTY:
                                    continue
                                prob = self._spread_prob(r, c, dr, dc)
                                if np.random.random() < prob:
                                    if self.hum_init[nr, nc] > 0:
                                        new_state[nr, nc]     = CellState.MOIST
                                        new_hum_timer[nr, nc] = 0
                                    else:
                                        new_state[nr, nc]      = CellState.BURNING
                                        new_burn_timer[nr, nc] = 0

                # ── Cel·la humida (escalfant-se) ─────────────
                elif s == CellState.MOIST:
                    new_hum_timer[r, c] += 1
                    if new_hum_timer[r, c] >= self.hum_init[r, c]:
                        new_state[r, c]      = CellState.BURNING
                        new_burn_timer[r, c] = 0

        self.state      = new_state
        self.hum        = new_hum
        self.veg        = new_veg
        self.burn_timer = new_burn_timer
        self.hum_timer  = new_hum_timer
        self.step      += 1
        self.history.append(self.state.copy())

    def stats(self):
        total   = self.rows * self.cols
        burning = int(np.sum(self.state == CellState.BURNING))
        moist   = int(np.sum(self.state == CellState.MOIST))
        burned  = int(np.sum(self.state == CellState.BURNED))
        safe    = total - burning - moist - burned
        return {"step": self.step, "burning": burning,
                "moist": moist, "burned": burned, "safe": safe, "total": total}

    def is_active(self):
        return (np.any(self.state == CellState.BURNING) or
                np.any(self.state == CellState.MOIST))


# ─────────────────────────────────────────────
#  VISUALITZACIÓ
# ─────────────────────────────────────────────
CMAP_COLORS = [
    "#639922",   # EMPTY   → verd (vegetació)
    "#F4C0D1",   # MOIST   → rosa (escalfant-se)
    "#EF9F27",   # BURNING → taronja (en flames)
    "#444441",   # BURNED  → gris fosc (cendra)
]
CMAP = mcolors.ListedColormap(CMAP_COLORS)
NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], CMAP.N)

LEGEND_PATCHES = [
    mpatches.Patch(color=CMAP_COLORS[0], label="Vegetació (sa)"),
    mpatches.Patch(color=CMAP_COLORS[1], label="Escalfant-se (humit)"),
    mpatches.Patch(color=CMAP_COLORS[2], label="En flames"),
    mpatches.Patch(color=CMAP_COLORS[3], label="Cremat"),
]


def plot_layers(hum_grid, veg_grid, state_grid, step=0):
    """Mostra les tres capes en una figura de 3 subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Simulació d'incendi forestal — Pas {step}", fontsize=14)

    im0 = axes[0].imshow(hum_grid, cmap="Blues", vmin=0, vmax=6)
    axes[0].set_title("Capa d'humitat (hores)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(veg_grid, cmap="Greens", vmin=0, vmax=15)
    axes[1].set_title("Capa de vegetació (hores)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    axes[2].imshow(state_grid, cmap=CMAP, norm=NORM)
    axes[2].set_title("Capa de propagació")
    axes[2].legend(handles=LEGEND_PATCHES, loc="upper right",
                   fontsize=7, framealpha=0.8)

    for ax in axes:
        ax.set_xlabel("columna")
        ax.set_ylabel("fila")

    plt.tight_layout()
    return fig


def animate_simulation(ca, steps, output_gif=None):
    """Crea una animació de la simulació."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ca.state, cmap=CMAP, norm=NORM, animated=True)
    ax.legend(handles=LEGEND_PATCHES, loc="upper right", fontsize=7, framealpha=0.8)
    title = ax.set_title("Pas 0")

    def update(frame):
        if ca.is_active():
            ca.advance()
        s = ca.stats()
        im.set_data(ca.state)
        title.set_text(
            f"Pas {s['step']}  |  Flames: {s['burning']}  "
            f"Escalfant: {s['moist']}  Cremat: {s['burned']}  Sa: {s['safe']}"
        )
        return [im, title]

    anim = FuncAnimation(fig, update, frames=steps, interval=300, blit=False, repeat=False)
    plt.tight_layout()

    if output_gif:
        anim.save(output_gif, writer="pillow", fps=3)
        print(f"  Animació guardada: {output_gif}")
    else:
        plt.show()
    return anim


def run_headless(ca, steps):
    """Executa la simulació sense GUI i mostra estadístiques per consola."""
    print(f"\n{'Pas':>5}  {'Flames':>7}  {'Escalfant':>9}  {'Cremat':>7}  {'Sa':>6}  {'% Cremat':>8}")
    print("-" * 55)
    for _ in range(steps):
        if not ca.is_active():
            break
        ca.advance()
        s = ca.stats()
        pct = s["burned"] / s["total"] * 100
        print(f"{s['step']:>5}  {s['burning']:>7}  {s['moist']:>9}  "
              f"{s['burned']:>7}  {s['safe']:>6}  {pct:>7.1f}%")
    print("-" * 55)
    s = ca.stats()
    print(f"\nResultat final — Pas {s['step']}:")
    print(f"  Cremat:     {s['burned']} cel·les ({s['burned']/s['total']*100:.1f}%)")
    print(f"  Sa:         {s['safe']}  cel·les ({s['safe']/s['total']*100:.1f}%)")


# ─────────────────────────────────────────────
#  PUNT D'ENTRADA
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Simulador d'incendi forestal — model m:n-CA (IDRISI32)"
    )
    parser.add_argument("--doc",    default="Initialize.doc",  help="Fitxer capçalera IDRISI32 (.doc)")
    parser.add_argument("--img",    default="Initialize.img",  help="Fitxer dades IDRISI32 (.img)")
    parser.add_argument("--dvc",    default="vegetation.dvc",  help="Fitxer capçalera vectorial (.dvc)")
    parser.add_argument("--vec",    default="vegetation.vec",  help="Fitxer polígons vectorials (.vec)")
    parser.add_argument("--rows",   type=int, default=20,      help="Files de la quadrícula (default: 20)")
    parser.add_argument("--cols",   type=int, default=20,      help="Columnes de la quadrícula (default: 20)")
    parser.add_argument("--ignite", type=int, nargs=2, default=[5, 5], metavar=("ROW","COL"),
                        help="Cel·la inicial del foc (default: 5 5)")
    parser.add_argument("--steps",  type=int, default=40,      help="Nombre màxim de passos (default: 40)")
    parser.add_argument("--wind",   type=int, nargs=2, default=[0, 0], metavar=("DR","DC"),
                        help="Direcció del vent com a (Δfila, Δcol), p.ex. 1 0 = sud (default: 0 0)")
    parser.add_argument("--animate", action="store_true",      help="Mostra animació interactiva")
    parser.add_argument("--save-gif", default=None,            help="Guarda animació com a GIF")
    parser.add_argument("--no-plot",  action="store_true",     help="No mostra cap finestra gràfica")
    args = parser.parse_args()

    # ── Comprova que existeixen els fitxers ──
    for f in [args.doc, args.img, args.dvc, args.vec]:
        if not os.path.isfile(f):
            print(f"  ERROR: No s'ha trobat el fitxer '{f}'")
            print("  Assegura't d'executar el script des del directori amb els fitxers,")
            print("  o passa les rutes amb --doc, --img, --dvc, --vec.")
            sys.exit(1)

    rows, cols = args.rows, args.cols

    print("=" * 55)
    print("  Simulador d'incendi forestal — m:n-CA^k")
    print("=" * 55)

    # ── Llegeix capa de humitat (IDRISI32) ──
    print("\n[1] Llegint capa d'humitat (IDRISI32)...")
    meta      = parse_idrisi_doc(args.doc)
    img_vals  = parse_idrisi_img_ascii(args.img)
    hum_grid  = build_humidity_grid(meta, img_vals, rows, cols)
    print(f"    Quadrícula: {rows}×{cols} | Valors únics .img: {set(img_vals)}")
    print(f"    Humitat min/max: {hum_grid.min()}/{hum_grid.max()} hores")

    # ── Llegeix capa de vegetació (IDRISI31 vectorial) ──
    print("\n[2] Llegint capa de vegetació (IDRISI31 vectorial)...")
    veg_meta  = parse_idrisi_doc(args.dvc)
    polygons  = parse_vec(args.vec)
    min_x = float(veg_meta.get("min. x", 0))
    max_x = float(veg_meta.get("max. x", 100))
    min_y = float(veg_meta.get("min. y", 0))
    max_y = float(veg_meta.get("max. y", 100))
    veg_grid  = build_vegetation_grid(polygons, rows, cols, min_x, max_x, min_y, max_y)
    print(f"    Polígons llegits: {len(polygons)}")
    for p in polygons:
        print(f"      id={p['id']} → {POLYGON_VEGETATION.get(p['id'], DEFAULT_VEGETATION)}h combustió, "
              f"{len(p['points'])} punts")
    print(f"    Vegetació min/max: {veg_grid.min()}/{veg_grid.max()} hores")

    # ── Construeix l'autòmat ──
    print(f"\n[3] Construint autòmat cel·lular ({rows}×{cols})...")
    wind = tuple(args.wind)
    ca   = WildfireCA(hum_grid, veg_grid, wind_dir=wind)

    if wind != (0, 0):
        dirs = {(1,0):"S", (-1,0):"N", (0,1):"E", (0,-1):"O",
                (1,1):"SE",(1,-1):"SO",(-1,1):"NE",(-1,-1):"NO"}
        print(f"    Vent: {dirs.get(wind, str(wind))}")

    # ── Mostra capes inicials ──
    if not args.no_plot and not args.animate and args.save_gif is None:
        print("\n[4] Mostrant capes inicials...")
        fig0 = plot_layers(hum_grid, veg_grid, ca.state, step=0)
        plt.savefig("capes_inicials.png", dpi=120, bbox_inches="tight")
        print("    Figura guardada: capes_inicials.png")
        plt.show()

    # ── Encén el foc ──
    ir, ic = args.ignite
    print(f"\n[5] Encenent foc a [{ir},{ic}]...")
    ca.ignite(ir, ic)

    # ── Simula ──
    if args.animate or args.save_gif:
        print(f"\n[6] Animant {args.steps} passos...")
        animate_simulation(ca, args.steps, output_gif=args.save_gif)
    elif args.no_plot:
        print(f"\n[6] Simulant {args.steps} passos (mode text)...")
        run_headless(ca, args.steps)
    else:
        print(f"\n[6] Simulant {args.steps} passos...")
        run_headless(ca, args.steps)

        # Figura final
        print("\n[7] Generant figura final...")
        fig1 = plot_layers(hum_grid, veg_grid, ca.state, step=ca.step)
        plt.savefig("resultat_final.png", dpi=120, bbox_inches="tight")
        print("    Figura guardada: resultat_final.png")
        plt.show()

    print("\nFet.")


if __name__ == "__main__":
    main()
