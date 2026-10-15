"""IA de vision (Qwen2.5-VL u otra) a traves de un servidor local compatible con OpenAI.

Pensado para LM Studio (http://localhost:1234/v1), que se encarga de usar la GPU (Vulkan).
Tambien sirve con Ollama (http://localhost:11434/v1) u otro servidor compatible.

Solo usa la biblioteca estandar + numpy/PIL, para que la interfaz pueda usarlo sin PyTorch.
"""
import base64
import io
import json
import re
import urllib.error
import urllib.request

import numpy as np

SERVIDOR = "http://localhost:1234/v1"
LADO = 28  # Qwen2.5-VL trabaja con trozos de 28 px: si la imagen ya es multiplo, no la reescala


class ErrorVision(RuntimeError):
    pass


def _peticion(servidor, ruta, datos=None, espera=300):
    url = servidor.rstrip("/") + ruta
    cuerpo = json.dumps(datos).encode() if datos is not None else None
    req = urllib.request.Request(url, data=cuerpo, headers={"Content-Type": "application/json"},
                                 method="POST" if datos is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=espera) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ErrorVision(f"No se pudo conectar con {url}: {e}. "
                          "Abre LM Studio y activa 'Servidor API local'.") from e


def modelos(servidor=SERVIDOR):
    return [m["id"] for m in _peticion(servidor, "/models").get("data", [])]


def elegir_modelo(servidor=SERVIDOR, preferido=None):
    """El modelo indicado, o el primero que parezca de vision (vl, vision, llava...)."""
    if preferido:
        return preferido
    lista = modelos(servidor)
    for m in lista:
        if re.search(r"vl|vision|llava|gemma-3|minicpm-v|moondream", m, re.I):
            return m
    if not lista:
        raise ErrorVision("El servidor no tiene ningun modelo. Descarga Qwen2.5-VL en LM Studio.")
    return lista[0]


def preparar(rgb, max_lado=896):
    """Reduce la imagen a un tamano multiplo de 28 (lo que usa Qwen) y la codifica en PNG/base64."""
    from PIL import Image
    im = Image.fromarray(rgb)
    escala = min(1.0, max_lado / max(im.size))
    w = max(LADO, int(im.size[0] * escala) // LADO * LADO)
    h = max(LADO, int(im.size[1] * escala) // LADO * LADO)
    im = im.resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), (w, h)


class IAVision:
    def __init__(self, servidor=SERVIDOR, modelo=None, max_lado=896):
        self.servidor = servidor
        self.modelo = elegir_modelo(servidor, modelo)
        self.max_lado = max_lado

    def preguntar(self, rgb, texto, max_tokens=400):
        """Pregunta sobre una imagen (array RGB). Devuelve (respuesta, tamano de la imagen enviada)."""
        b64, tam = preparar(rgb, self.max_lado)
        datos = {
            "model": self.modelo,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": texto},
            ]}],
        }
        r = _peticion(self.servidor, "/chat/completions", datos)
        try:
            return r["choices"][0]["message"]["content"] or "", tam
        except (KeyError, IndexError) as e:
            raise ErrorVision(f"Respuesta inesperada del servidor: {str(r)[:300]}") from e

    def describir(self, rgb):
        texto, _ = self.preguntar(rgb, "Esto es una captura de un videojuego. Describe en espanol, en pocas "
                                       "frases, que se ve: el entorno, los objetos y cualquier cosa util para jugar.")
        return texto

    def localizar(self, rgb, objeto):
        """Cajas (x1, y1, x2, y2) en pixeles de la imagen ORIGINAL donde aparece 'objeto'."""
        texto, (w, h) = self.preguntar(
            rgb,
            f"Locate every '{objeto}' in this video game screenshot. "
            'Output only JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
            "If there is none, output [].", max_tokens=600)
        cajas = leer_cajas(texto)
        alto, ancho = rgb.shape[:2]
        # Qwen2.5-VL da pixeles de la imagen que recibio; algunos modelos dan 0..1000 o 0..1
        maximo = max((max(c) for c in cajas), default=0)
        if maximo <= 1.0:
            ref = (1.0, 1.0)
        elif maximo > max(w, h) * 1.05 and maximo <= 1000:
            ref = (1000.0, 1000.0)
        else:
            ref = (float(w), float(h))
        sx, sy = ancho / ref[0], alto / ref[1]
        salida = []
        for x1, y1, x2, y2 in cajas:
            x1, x2 = sorted((x1 * sx, x2 * sx))
            y1, y2 = sorted((y1 * sy, y2 * sy))
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(ancho, int(x2)), min(alto, int(y2))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                salida.append((x1, y1, x2, y2))
        return salida, texto


def leer_cajas(texto):
    """Saca las cajas [x1, y1, x2, y2] de la respuesta (JSON, a veces dentro de ```json ... ```)."""
    cajas = []
    m = re.search(r"\[.*\]", texto, re.S)
    if m:
        try:
            datos = json.loads(m.group(0))
            for d in datos if isinstance(datos, list) else [datos]:
                caja = d.get("bbox_2d") or d.get("bbox") or d.get("box") if isinstance(d, dict) else d
                if isinstance(caja, (list, tuple)) and len(caja) == 4:
                    cajas.append([float(v) for v in caja])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if not cajas:  # respuesta no valida como JSON: buscar grupos de 4 numeros
        for g in re.findall(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", texto):
            cajas.append([float(v) for v in g])
    return cajas


def recorte_interior(rgb, caja, margen=0.2):
    """El centro de la caja (sin los bordes, que suelen tener fondo)."""
    x1, y1, x2, y2 = caja
    dx, dy = int((x2 - x1) * margen), int((y2 - y1) * margen)
    trozo = rgb[y1 + dy:y2 - dy, x1 + dx:x2 - dx]
    return trozo if trozo.size else rgb[y1:y2, x1:x2]


def dibujar_cajas(rgb, cajas, ruta):
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(rgb))
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in cajas:
        d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=3)
    im.save(ruta)
