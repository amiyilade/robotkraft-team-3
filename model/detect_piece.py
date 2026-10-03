import sys
import cv2
import numpy as np

from camera_utils import ZMQCamera, white_balance

SAT_MIN = 80        # seuil de saturation : les pièces sont colorées, la table non
VAL_MIN = 50
MIN_AREA = 300      # ignore le bruit (pixels)
CUBE_AREA = None    # aire d'un cube en pixels, à calibrer (touche C)


def segment(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, SAT_MIN, VAL_MIN), (179, 255, 255))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def classify(cnt, cube_area):
    """Retourne (forme, nb_cases) pour un contour"""
    area = cv2.contourArea(cnt)
    cells = max(1, round(area / cube_area))
    if cells == 1:
        return "cube", 1
    if cells != 4:
        return f"inconnu({cells})", cells

    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    long_side, short_side = max(w, h), min(w, h)
    ratio = long_side / short_side
    fill = area / (w * h)

    if ratio > 2.8:
        return "I", 4
    if ratio < 1.3 and fill > 0.85:
        return "carre", 4

    # L ou T : on regarde où est le centre de gravité le long du grand axe
    m = cv2.moments(cnt)
    gx, gy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    theta = np.deg2rad(angle if w >= h else angle + 90)
    axis = np.array([np.cos(theta), np.sin(theta)])
    offset = abs(np.dot(np.array([gx - cx, gy - cy]), axis)) / long_side
    return ("L" if offset > 0.04 else "T"), 4


def detect(frame, cube_area):
    mask = segment(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        shape, cells = classify(cnt, cube_area)
        (cx, cy), _, angle = cv2.minAreaRect(cnt)
        pieces.append({"shape": shape, "cells": cells, "center": (cx, cy), "angle": angle, "contour": cnt})
    return pieces, mask


def main():
    global CUBE_AREA
    cap = ZMQCamera().open()
    print("C = calibrer (UN SEUL cube visible), Q = quitter")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = white_balance(frame)
        mask = segment(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                CUBE_AREA = cv2.contourArea(max(cnts, key=cv2.contourArea))
                print(f"Aire d'un cube : {CUBE_AREA:.0f} px")
        if key in (ord("q"), 27):
            break

        if CUBE_AREA:
            pieces, _ = detect(frame, CUBE_AREA)
            for p in pieces:
                cv2.drawContours(frame, [p["contour"]], -1, (0, 255, 0), 2)
                x, y = map(int, p["center"])
                cv2.putText(frame, p["shape"], (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Duplo", frame)
        cv2.imshow("mask", mask)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()