import pygame
import numpy as np


# ── Condició frontera: zeros (les cèl·lules fora del grid valen 0) ──────────

def get_rule(rule_number):
    """Converteix el número de regla (0-255) en una llista de 8 bits."""
    return [int(x) for x in f"{rule_number:08b}"]


def next_cell(l, c, r, rule):
    """Retorna el valor de la cèl·lula al següent pas."""
    return rule[7 - ((l << 2) | (c << 1) | r)]


def evolve(state, rule_bits):
    """Calcula el següent estat del CA (frontera de zeros)."""
    cols = len(state)
    new_state = np.zeros(cols, dtype=int)
    for i in range(cols):
        l = state[i - 1] if i > 0 else 0          # frontera esquerra = 0
        c = state[i]
        r = state[i + 1] if i < cols - 1 else 0   # frontera dreta = 0
        new_state[i] = next_cell(l, c, r, rule_bits)
    return new_state


# ── Gra guixut K=2 ──────────────────────────────────────────────────────────

def gra_guixut(history, K=2):
    """
    Renormalitza l'historial agrupant K cèl·lules per bloc (vot de majoria).
    Retorna el nou historial renormalitzat i la regla Wolfram inferida (0-255).
    """
    cg_history = []
    for state in history:
        n_blocs = len(state) // K
        blocs = state[:n_blocs * K].reshape(n_blocs, K)
        cg_state = (blocs.sum(axis=1) > K / 2).astype(int)
        cg_history.append(cg_state)

    # Inferència de la regla a partir de les transicions observades
    pats = ['111', '110', '101', '100', '011', '010', '001', '000']
    comptes = {p: {0: 0, 1: 0} for p in pats}
    n_blocs = len(cg_history[0])

    for g in range(len(cg_history) - 1):
        for i in range(n_blocs):
            l = cg_history[g][(i - 1) % n_blocs]
            c = cg_history[g][i]
            r = cg_history[g][(i + 1) % n_blocs]
            comptes[f"{l}{c}{r}"][int(cg_history[g + 1][i])] += 1

    inferida = {p: (1 if comptes[p][1] >= comptes[p][0] else 0) for p in pats}
    regla_inf = sum(inferida[p] << (7 - i) for i, p in enumerate(pats))

    total   = sum(comptes[p][0] + comptes[p][1] for p in pats)
    encerts = sum(comptes[p][inferida[p]] for p in pats)
    precisio = encerts / total * 100 if total else 0.0

    return cg_history, regla_inf, precisio


# ── Visualització pygame ─────────────────────────────────────────────────────

def draw_history(screen, history, x_offset, cols, cell_size, color):
    for gen, state in enumerate(history):
        for i in range(min(cols, len(state))):
            if state[i]:
                pygame.draw.rect(screen, color,
                                 (x_offset + i * cell_size,
                                  gen * cell_size,
                                  cell_size, cell_size))


def draw_label(screen, font, text, x_offset, color=(255, 255, 255)):
    screen.blit(font.render(text, True, color), (x_offset + 8, 8))


def draw_divider(screen, x, height):
    pygame.draw.line(screen, (100, 100, 100), (x, 0), (x, height))


# ── Regla simple ─────────────────────────────────────────────────────────────

def run_single_rule(rule_number, width=900, height=400, cell_size=2):
    """
    Mostra tres panells:
      1. CA original (regla donada)
      2. CA de gra guixut K=2
      3. Diferències entre original mostrejat i gra guixut
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Regla {rule_number} — Gra Guixut K=2")
    font  = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    rows      = height // cell_size
    panel_w   = width // 3
    cols      = panel_w // cell_size
    rule_bits = get_rule(rule_number)

    state = np.zeros(cols, dtype=int)
    state[cols // 2] = 1

    history = [state.copy()]
    generation = 1
    running = True

    while running:
        screen.fill((20, 20, 20))

        # Calcula gra guixut i diferències
        cg_history, regla_inf, precisio = gra_guixut(history, K=2)
        cg_cols = len(cg_history[0])

        # Panell 1: CA original
        draw_history(screen, history, 0, cols, cell_size, (255, 255, 255))
        draw_label(screen, font, f"Regla {rule_number}", 0)

        # Panell 2: Gra guixut
        draw_history(screen, cg_history, panel_w, cg_cols, cell_size * 2, (100, 200, 255))
        draw_label(screen, font,
                   f"Gra guixut K=2  →  regla inferida: {regla_inf}  ({precisio:.0f}%)",
                   panel_w, (100, 200, 255))

        # Panell 3: Diferències (original mostrejat cada 2 vs gra guixut)
        for gen, (orig, cg) in enumerate(zip(history, cg_history)):
            orig_sampled = orig[:cg_cols * 2:2][:cg_cols]
            for i in range(min(cg_cols, len(orig_sampled))):
                if orig_sampled[i] != cg[i]:
                    pygame.draw.rect(screen, (255, 80, 80),
                                     (panel_w * 2 + i * cell_size * 2,
                                      gen * cell_size,
                                      cell_size * 2, cell_size))
        draw_label(screen, font, "Diferències", panel_w * 2, (255, 80, 80))

        draw_divider(screen, panel_w, height)
        draw_divider(screen, panel_w * 2, height)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if generation < rows:
            state = evolve(state, rule_bits)
            history.append(state.copy())
            generation += 1

    pygame.quit()


# ── Combinació de dues regles ────────────────────────────────────────────────

def run_combination(rule1, rule2, width=1200, height=400, cell_size=2):
    """
    Mostra quatre panells:
      1. CA regla1 (vermell)
      2. CA regla2 (blau)
      3. Combinació (màxim, lila)
      4. Gra guixut K=2 de la combinació (verd)
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Combinació regles {rule1}+{rule2} — Gra Guixut K=2")
    font  = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    rows    = height // cell_size
    panel_w = width // 4
    cols    = panel_w // cell_size

    rule1_bits = get_rule(rule1)
    rule2_bits = get_rule(rule2)

    state1 = np.zeros(cols, dtype=int)
    state2 = np.zeros(cols, dtype=int)
    state1[cols // 2] = state2[cols // 2] = 1

    history1    = [state1.copy()]
    history2    = [state2.copy()]
    history_comb = [np.maximum(state1, state2).copy()]
    generation  = 1
    running     = True

    while running:
        screen.fill((20, 20, 20))

        cg_history, regla_inf, precisio = gra_guixut(history_comb, K=2)
        cg_cols = len(cg_history[0])

        draw_history(screen, history1,    0,          cols,    cell_size,     (255, 80,  80))
        draw_history(screen, history2,    panel_w,    cols,    cell_size,     (80,  80,  255))
        draw_history(screen, history_comb, panel_w*2, cols,    cell_size,     (200, 0,   255))
        draw_history(screen, cg_history,  panel_w*3,  cg_cols, cell_size * 2, (80,  220, 120))

        draw_label(screen, font, f"Regla {rule1}",    0,          (255, 80,  80))
        draw_label(screen, font, f"Regla {rule2}",    panel_w,    (80,  80,  255))
        draw_label(screen, font, "Combinació",         panel_w*2,  (200, 0,   255))
        draw_label(screen, font,
                   f"Gra guixut K=2 → {regla_inf} ({precisio:.0f}%)",
                   panel_w*3, (80, 220, 120))

        for x in [panel_w, panel_w*2, panel_w*3]:
            draw_divider(screen, x, height)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if generation < rows:
            state1 = evolve(state1, rule1_bits)
            state2 = evolve(state2, rule2_bits)
            combined = np.maximum(state1, state2)

            history1.append(state1.copy())
            history2.append(state2.copy())
            history_comb.append(combined.copy())
            generation += 1

    pygame.quit()


# ── Execució ─────────────────────────────────────────────────────────────────

run_single_rule(rule_number=124)
run_combination(rule1=110, rule2=124)