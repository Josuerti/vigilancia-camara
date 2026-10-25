# Vigilancia con detección de personas — Uniarch Uho-S2E

Sistema de video-analítica para tu cámara IP: lee el stream **RTSP**, detecta
**personas** en tiempo real con YOLOv8, registra eventos con hora, y guarda
**snapshots** y **clips** de video cuando aparece alguien. Incluye zonas de
interés y reconexión automática.

## Puesta en marcha

1. **Credenciales.** Copia `.env.example` a `.env` y pon el usuario y la
   contraseña de la cámara (los que usas en `http://192.168.100.45`):

   ```
   CAM_USER=admin
   CAM_PASS=tu_clave_real
   ```

2. **Probar la conexión** al stream:

   ```bat
   run.bat --probar-rtsp
   ```

   Si falla, prueba cambiando `CAM_RTSP_PATH` en `.env` (el script ya intenta
   las rutas comunes de Uniview/Uniarch automáticamente).

3. **Arrancar la vigilancia** (con ventana en vivo):

   ```bat
   run.bat
   ```

   Pulsa **q** en la ventana para salir. Sin pantalla (modo servidor):

   ```bat
   run.bat --sin-ventana
   ```

## Configuración (`config.yaml`)

- `modelo.dispositivo`: `cpu` o `cuda` (GPU).
- `modelo.confianza`: umbral de detección (0–1).
- `modelo.clases`: `[0]` = solo personas. Añade `2` (carro), `16` (perro)…
- `eventos.cooldown_seg`: tiempo mínimo entre eventos.
- `eventos.clip_seg`: duración de los clips.
- `zona`: polígono (fracciones 0–1) para vigilar solo una región.

## Salidas

Todo se guarda en `eventos/`:
- `eventos.csv` — registro con fecha, nº de personas y confianza.
- `persona_AAAAMMDD_HHMMSS.jpg` — foto del momento.
- `persona_AAAAMMDD_HHMMSS.mp4` — clip con 2 s previos de contexto.

## Roadmap

- **Fase 2 — qué hace la persona:** pose (YOLOv8-pose) para distinguir de pie /
  agachado / caído (detección de caídas) y merodeo.
- **Alertas:** notificación a Telegram o correo al detectar persona.
- **GPU:** mover la inferencia a la RTX 5060 para tiempo real a full resolución.
- **Arranque automático:** tarea programada de Windows al iniciar sesión.
