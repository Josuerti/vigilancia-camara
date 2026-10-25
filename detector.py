"""
Vigilancia con detección de personas para cámara Uniarch/Uniview (RTSP).

Lee el stream de la cámara, detecta personas en tiempo real con YOLOv8,
registra eventos (hora, confianza), guarda snapshots y clips de video, y
permite limitar la vigilancia a una zona del cuadro.

Uso:
    python detector.py                 # usa config.yaml
    python detector.py --sin-ventana   # modo servidor (sin pantalla)
    python detector.py --probar-rtsp   # solo prueba la conexión RTSP

Las credenciales se leen de .env ; los ajustes de config.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    print("Falta ultralytics. Instala con: uv pip install ultralytics")
    raise

RAIZ = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
def cargar_env(ruta: Path) -> dict:
    """Parser mínimo de .env (sin dependencias externas)."""
    datos: dict[str, str] = {}
    if not ruta.exists():
        return datos
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        datos[clave.strip()] = valor.strip().strip('"').strip("'")
    return datos


def cargar_config() -> tuple[dict, dict]:
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    env = cargar_env(RAIZ / ".env")
    if not env.get("CAM_PASS") or env["CAM_PASS"] == "TU_CONTRASEÑA_AQUI":
        print("⚠️  Edita el archivo .env con el usuario y contraseña de tu cámara.")
    return cfg, env


def construir_urls(env: dict) -> list[str]:
    """URL principal + alternativas comunes de Uniview/Uniarch por si falla."""
    user = env.get("CAM_USER", "admin")
    pw = env.get("CAM_PASS", "")
    ip = env.get("CAM_IP", "192.168.100.45")
    port = env.get("CAM_PORT", "554")
    cred = f"{user}:{pw}@" if pw else f"{user}@"
    base = f"rtsp://{cred}{ip}:{port}"
    rutas = [env.get("CAM_RTSP_PATH", "/media/video1")]
    # Rutas típicas de Uniview/Uniarch y ONVIF genéricas (sin duplicar).
    for r in ("/media/video1", "/media/video2",
              "/unicast/c1/s0/live", "/unicast/c1/s1/live",
              "/cam/realmonitor?channel=1&subtype=0", "/live/ch0", "/0"):
        if r not in rutas:
            rutas.append(r)
    return [base + r for r in rutas]


# --------------------------------------------------------------------------- #
# Captura RTSP
# --------------------------------------------------------------------------- #
def abrir_stream(urls: list[str], transporte: str):
    """Intenta cada URL y devuelve (captura, url_ok) o (None, None)."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transporte}|stimeout;5000000"
    )
    for url in urls:
        segura = url.split("@")[-1]  # no imprimir credenciales
        print(f"→ Probando RTSP …@{segura}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"✅ Conectado: …@{segura}")
                return cap, url
        cap.release()
    return None, None


# --------------------------------------------------------------------------- #
# Zona de interés
# --------------------------------------------------------------------------- #
def poligono_px(zona: list, w: int, h: int):
    if not zona:
        return None
    return np.array([[int(x * w), int(y * h)] for x, y in zona], dtype=np.int32)


def dentro_de_zona(poli, punto) -> bool:
    if poli is None:
        return True
    return cv2.pointPolygonTest(poli, punto, False) >= 0


# --------------------------------------------------------------------------- #
# Registro de eventos
# --------------------------------------------------------------------------- #
class Registro:
    def __init__(self, carpeta: Path):
        self.carpeta = carpeta
        carpeta.mkdir(parents=True, exist_ok=True)
        self.csv_path = carpeta / "eventos.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["fecha_hora", "personas", "confianza_max", "snapshot", "clip"]
                )

    def anotar(self, n, conf, snap, clip):
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [datetime.now().isoformat(timespec="seconds"),
                 n, f"{conf:.2f}", snap or "", clip or ""]
            )


# --------------------------------------------------------------------------- #
# Bucle principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Detección de personas por RTSP")
    ap.add_argument("--sin-ventana", action="store_true",
                    help="No mostrar ventana (modo servidor).")
    ap.add_argument("--probar-rtsp", action="store_true",
                    help="Solo probar la conexión y salir.")
    args = ap.parse_args()

    cfg, env = cargar_config()
    urls = construir_urls(env)

    cap, url_ok = abrir_stream(urls, cfg["stream"]["transporte"])
    if cap is None:
        print("❌ No se pudo abrir ninguna ruta RTSP. Revisa IP, usuario y "
              "contraseña en .env, o la ruta CAM_RTSP_PATH.")
        sys.exit(1)
    if args.probar_rtsp:
        print("✅ Prueba de RTSP exitosa.")
        cap.release()
        return

    # Modelo
    modelo = YOLO(cfg["modelo"]["pesos"])
    dispositivo = cfg["modelo"]["dispositivo"]
    clases = cfg["modelo"]["clases"]
    conf_min = float(cfg["modelo"]["confianza"])
    print(f"🧠 Modelo {cfg['modelo']['pesos']} en {dispositivo}")

    mostrar = cfg["ui"]["mostrar_ventana"] and not args.sin_ventana
    reg = Registro(RAIZ / cfg["eventos"]["carpeta"])

    ancho_proc = int(cfg["stream"]["ancho_proceso"])
    cooldown = float(cfg["eventos"]["cooldown_seg"])
    guardar_snap = cfg["eventos"]["guardar_snapshot"]
    guardar_clip = cfg["eventos"]["guardar_clip"]
    clip_seg = float(cfg["eventos"]["clip_seg"])

    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 15.0
    buffer_pre = deque(maxlen=int(fps_cam * 2))  # 2 s previos al evento
    ultimo_evento = 0.0
    grabando_hasta = 0.0
    escritor_clip = None
    clip_actual = None
    poli = None

    t0, frames, fps_vis = time.time(), 0, 0.0
    print("▶️  Vigilancia iniciada. Ctrl+C para detener.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️  Stream caído, reintentando…")
                cap.release()
                time.sleep(cfg["stream"]["reconectar_seg"])
                cap, url_ok = abrir_stream(urls, cfg["stream"]["transporte"])
                if cap is None:
                    time.sleep(cfg["stream"]["reconectar_seg"])
                continue

            h0, w0 = frame.shape[:2]
            if poli is None:
                poli = poligono_px(cfg.get("zona", []), w0, h0)

            # Redimensionar para inferir más rápido
            if ancho_proc and w0 > ancho_proc:
                escala = ancho_proc / w0
                proc = cv2.resize(frame, (ancho_proc, int(h0 * escala)))
            else:
                escala, proc = 1.0, frame

            res = modelo.predict(proc, conf=conf_min, classes=clases,
                                 device=dispositivo, verbose=False)[0]

            personas, conf_max = 0, 0.0
            for box in res.boxes:
                x1, y1, x2, y2 = (box.xyxy[0] / escala).tolist()
                c = float(box.conf[0])
                pies = (int((x1 + x2) / 2), int(y2))  # punto base
                if not dentro_de_zona(poli, pies):
                    continue
                personas += 1
                conf_max = max(conf_max, c)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 0, 255), 2)
                cv2.putText(frame, f"persona {c:.2f}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if poli is not None and cfg["ui"]["dibujar_zona"]:
                cv2.polylines(frame, [poli], True, (0, 255, 255), 2)

            # HUD
            ahora = time.time()
            frames += 1
            if ahora - t0 >= 1.0:
                fps_vis, frames, t0 = frames / (ahora - t0), 0, ahora
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS {fps_vis:.1f}  personas {personas}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            buffer_pre.append(frame.copy())

            # ---- Evento ----
            if personas > 0 and (ahora - ultimo_evento) > cooldown:
                ultimo_evento = ahora
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap = clip = None
                if guardar_snap:
                    snap = str(reg.carpeta / f"persona_{ts}.jpg")
                    cv2.imwrite(snap, frame)
                if guardar_clip:
                    clip = str(reg.carpeta / f"persona_{ts}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    escritor_clip = cv2.VideoWriter(
                        clip, fourcc, fps_cam, (w0, h0))
                    for f_prev in buffer_pre:      # incluir contexto previo
                        escritor_clip.write(f_prev)
                    clip_actual = clip
                    grabando_hasta = ahora + clip_seg
                reg.anotar(personas, conf_max, snap, clip)
                print(f"🚨 {datetime.now():%H:%M:%S}  {personas} persona(s) "
                      f"(conf {conf_max:.2f})  → {Path(snap).name if snap else '-'}")

            # ---- Grabación de clip en curso ----
            if escritor_clip is not None:
                escritor_clip.write(frame)
                if ahora >= grabando_hasta:
                    escritor_clip.release()
                    escritor_clip = None
                    print(f"   🎬 Clip guardado: {Path(clip_actual).name}")

            if mostrar:
                cv2.imshow("Vigilancia - [q] para salir", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n⏹️  Detenido por el usuario.")
    finally:
        if escritor_clip is not None:
            escritor_clip.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"📁 Eventos en: {reg.carpeta}")


if __name__ == "__main__":
    main()
