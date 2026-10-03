#!/usr/bin/env python3
"""
MIX : calibration des mélanges de colorants et recherche de recette.

Principe : on ne calcule pas le mélange (système soustractif, colorants
réels), on le MESURE. On photographie chaque recette entière à côté du carré
blanc, on normalise par le blanc, on passe en Lab, puis on cherche la recette
mesurée la plus proche de la cible.

Nom des photos = la recette : cyan2_magenta1_jaune0_eau1.jpg
(ingrédient + nombre de tubes, séparés par _ ; les noms sont libres)

Commandes :
  check    dessine les zones (verre, blanc) sur une photo pour les vérifier
  measure  mesure toutes les photos -> recettes.json
  pick     sélectionne les zones à la souris -> affiche --glass et --white
  snap     sauve une frame du flux caméra (pour régler les zones)
  record   mesure une recette depuis le flux caméra -> recettes.json
  live     suit le mélange en direct (recette proche, écart à la cible)
  solve    cible RGB (niveau avancé) -> meilleures recettes
  palette  cible dans une liste de couleurs (niveau simple) + test de classement

Dépendances : opencv-python-headless, numpy (déjà installés avec lerobot).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

D65 = np.array([0.95047, 1.0, 1.08883])
RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])


# ---------- couleur ----------
def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_lab(lin):
    xyz = (RGB2XYZ @ lin) / D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def hex_to_lab(code):
    code = code.strip().lstrip("#")
    rgb = np.array([int(code[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return linear_to_lab(srgb_to_linear(rgb))


def delta_e(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))  # CIE76


# ---------- image ----------
def parse_roi(text):
    x, y, w, h = (int(v) for v in text.split(","))
    return x, y, w, h


def roi_rgb(img, roi):
    x, y, w, h = roi
    patch = img[y:y + h, x:x + w]
    if patch.size == 0:
        sys.exit(f"Zone {roi} hors de l'image ({img.shape[1]}x{img.shape[0]}).")
    # médiane : robuste aux reflets sur le verre
    return np.median(patch.reshape(-1, 3), axis=0)[::-1] / 255.0  # BGR -> RGB


def measure_image(path, glass, white):
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Image illisible : {path}")
    return measure_frame(img, glass, white)


def measure_frame(img, glass, white):
    g = srgb_to_linear(roi_rgb(img, glass))
    w = srgb_to_linear(roi_rgb(img, white))
    lin = np.clip(g / np.maximum(w, 1e-6), 0.0, 1.0)  # le blanc devient (1,1,1)
    return linear_to_lab(lin)


def parse_recipe(stem):
    pairs = re.findall(r"([^\W\d_]+)(\d+)", stem)
    if not pairs:
        sys.exit(f"Nom de fichier sans recette : {stem} (ex : cyan2_magenta1_eau1)")
    return {name: int(n) for name, n in pairs}


def recipe_str(recipe):
    return " + ".join(f"{n} {name}" for name, n in recipe.items() if n > 0) or "vide"


# ---------- flux caméra ----------
def open_source(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la source vidéo : {source} "
                 "(sous WSL : usbipd attach + ls /dev/video*)")
    for _ in range(10):  # laisse l'auto-exposition se stabiliser
        cap.read()
    return cap


def grab_labs(source, glass, white, frames):
    """Mesure sur plusieurs frames, médiane -> bruit capteur réduit."""
    cap = open_source(source)
    labs = []
    for _ in range(frames):
        ok, img = cap.read()
        if ok:
            labs.append(measure_frame(img, glass, white))
    cap.release()
    if not labs:
        sys.exit("Aucune frame lue.")
    return np.median(np.array(labs), axis=0)


# ---------- commandes ----------
def cmd_check(a):
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"Image illisible : {a.image}")
    for roi, color in ((parse_roi(a.glass), (0, 0, 255)), (parse_roi(a.white), (0, 255, 0))):
        x, y, w, h = roi
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    cv2.imwrite(a.out, img)
    print(f"Écrit {a.out} : rouge = verre, vert = carré blanc. Ouvre-le et ajuste.")


def cmd_measure(a):
    glass, white = parse_roi(a.glass), parse_roi(a.white)
    files = sorted(p for p in Path(a.folder).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        sys.exit("Aucune photo trouvée.")
    table = []
    for p in files:
        if not re.search(r"[^\W\d_]+\d+", p.stem):
            print(f"Ignoré (pas une recette) : {p.name}")
            continue
        lab = measure_image(p, glass, white)
        table.append({"file": p.name, "recipe": parse_recipe(p.stem), "lab": lab.tolist()})
        print(f"{p.name:40s} Lab = {np.round(lab, 1)}")
    Path(a.out).write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"\n{len(table)} recettes écrites dans {a.out}")


def cmd_pick(a):
    """Sélection à la souris des zones (nécessite opencv-python, pas -headless)."""
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"Image illisible : {a.image}")
    scale = min(1.0, 1280 / img.shape[1])  # réduit l'affichage si l'image est trop grande
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    print("Dessine un rectangle à la souris, puis Entrée (ou Espace). C pour annuler.")
    try:
        g = cv2.selectROI("1) LIQUIDE dans le verre, puis Entree", view, showCrosshair=False)
        w = cv2.selectROI("2) CARRE BLANC, puis Entree", view, showCrosshair=False)
    except cv2.error:
        sys.exit("Pas d'interface graphique : installe 'opencv-python' (pas -headless) "
                 "et lance depuis Windows, ou utilise 'check' avec des coordonnées à la main.")
    cv2.destroyAllWindows()
    if 0 in g[2:] or 0 in w[2:]:
        sys.exit("Sélection annulée.")
    fmt = lambda r: ",".join(str(int(round(v / scale))) for v in r)
    print(f"\nCopie ces options dans tes commandes :\n  --glass {fmt(g)} --white {fmt(w)}")


def cmd_snap(a):
    cap = open_source(a.source)
    ok, img = cap.read()
    cap.release()
    if not ok:
        sys.exit("Aucune frame lue.")
    cv2.imwrite(a.out, img)
    print(f"Frame écrite dans {a.out} ({img.shape[1]}x{img.shape[0]}). "
          "Utilise-la avec 'check' pour trouver les zones x,y,w,h.")


def cmd_record(a):
    lab = grab_labs(a.source, parse_roi(a.glass), parse_roi(a.white), a.frames)
    path = Path(a.table)
    table = json.loads(path.read_text()) if path.exists() else []
    recipe = parse_recipe(a.recipe)
    table = [e for e in table if e["recipe"] != recipe]  # remplace si déjà mesurée
    table.append({"file": f"live:{a.recipe}", "recipe": recipe, "lab": lab.tolist()})
    path.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"{recipe_str(recipe)} -> Lab = {np.round(lab, 1)} ({len(table)} recettes dans {a.table})")


def cmd_live(a):
    import time
    table = load_table(a.table)
    glass, white = parse_roi(a.glass), parse_roi(a.white)
    cap = open_source(a.source)
    target = hex_to_lab(a.hex) if a.hex else None
    print("Ctrl+C pour arrêter.")
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            lab = measure_frame(img, glass, white)
            d, e = nearest(table, lab, 1)[0]
            line = f"Lab={np.round(lab, 1)}  recette proche : {recipe_str(e['recipe'])} (dE={d:.1f})"
            if target is not None:
                line += f"  | écart à la cible : dE={delta_e(lab, target):.1f}"
            print(line)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def load_table(path):
    return json.loads(Path(path).read_text())


def nearest(table, lab, k=3):
    scored = sorted(((delta_e(e["lab"], lab), e) for e in table), key=lambda t: t[0])
    return scored[:k]


def cmd_solve(a):
    table = load_table(a.table)
    for dist, e in nearest(table, hex_to_lab(a.hex), a.top):
        print(f"dE={dist:5.1f}  {recipe_str(e['recipe']):40s} ({e['file']})")


def cmd_palette(a):
    table = load_table(a.table)
    palette = [c.strip() for c in a.colors.split(",")]
    labs = [hex_to_lab(c) for c in palette]
    for i, code in enumerate(palette):
        dist, best = nearest(table, labs[i], 1)[0]
        # le jury classe : parmi la liste, quelle couleur est la plus proche du mélange ?
        ranks = [delta_e(best["lab"], l) for l in labs]
        won = int(np.argmin(ranks)) == i
        status = "OK" if won else f"RISQUE (le jury verrait plutôt {palette[int(np.argmin(ranks))]})"
        print(f"{code}: {recipe_str(best['recipe']):38s} dE={dist:5.1f}  {status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check")
    p.add_argument("image")
    p.add_argument("--glass", required=True, help="x,y,w,h du liquide dans le verre")
    p.add_argument("--white", required=True, help="x,y,w,h du carré blanc")
    p.add_argument("--out", default="check.jpg")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("measure")
    p.add_argument("folder")
    p.add_argument("--glass", required=True)
    p.add_argument("--white", required=True)
    p.add_argument("--out", default="recettes.json")
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("pick", help="sélectionne les zones à la souris et affiche --glass/--white")
    p.add_argument("image")
    p.set_defaults(fn=cmd_pick)

    p = sub.add_parser("snap", help="sauve une frame du flux caméra")
    p.add_argument("--source", default="0", help="index caméra (0, 1...) ou fichier/URL vidéo")
    p.add_argument("--out", default="snap.jpg")
    p.set_defaults(fn=cmd_snap)

    p = sub.add_parser("record", help="mesure une recette depuis le flux et l'ajoute à la table")
    p.add_argument("--source", default="0")
    p.add_argument("--recipe", required=True, help="ex : cyan2_magenta1_eau1")
    p.add_argument("--glass", required=True)
    p.add_argument("--white", required=True)
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--table", default="recettes.json")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("live", help="suit le mélange en direct (recette proche, écart à la cible)")
    p.add_argument("--source", default="0")
    p.add_argument("--glass", required=True)
    p.add_argument("--white", required=True)
    p.add_argument("--table", default="recettes.json")
    p.add_argument("--hex", help="cible optionnelle, ex : #D2691E")
    p.set_defaults(fn=cmd_live)

    p = sub.add_parser("solve")
    p.add_argument("--table", default="recettes.json")
    p.add_argument("--hex", required=True, help="cible, ex : #D2691E")
    p.add_argument("--top", type=int, default=3)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("palette")
    p.add_argument("--table", default="recettes.json")
    p.add_argument("--colors", required=True, help="liste hex séparée par des virgules")
    p.set_defaults(fn=cmd_palette)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()