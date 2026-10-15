"""Usa una IA de vision (Qwen2.5-VL en LM Studio) para entender la pantalla del juego.

Necesita LM Studio con un modelo de vision descargado y el "Servidor API local" activado
(http://localhost:1234/v1). LM Studio usa la GPU por su cuenta (motor Vulkan).

  python ver_pantalla.py --describir --region auto
      dice que se ve en el juego
  python ver_pantalla.py --nombre Minecraft --localizar madera --region auto
      busca la madera y guarda una imagen con cajas rojas (vision_resultado.png)
  python ver_pantalla.py --nombre Minecraft --ensenar madera --region auto
      la IA de vision encuentra la madera y se la ENSENA al detector rapido de la IA de juego
  python ver_pantalla.py --nombre Minecraft --ensenar madera --repetir 10 --cada 20
      lo repite cada 20 s mientras juegas o exploras (mas ejemplos = reconoce mejor)
"""
import argparse
import time

from juegos.objetos import anadir_ejemplo, capturar_zona
from juegos.vision import SERVIDOR, ErrorVision, IAVision, dibujar_cajas, recorte_interior
from juegos.ventana import ventana_activa


def zona_de(region):
    if region.strip().lower() == "auto":
        titulo, zona = ventana_activa()
        if zona and zona[2] >= 50 and zona[3] >= 50:
            print(f"Juego detectado: '{titulo}'")
            return zona
        print("Aviso: no se detecto la ventana del juego; se usa la pantalla completa")
        region = ""
    if region.strip():
        return tuple(int(v) for v in region.split(","))
    import mss
    with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
        m = sct.monitors[1]
    return m["left"], m["top"], m["width"], m["height"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", default="juego", help="nombre del juego (para --ensenar)")
    p.add_argument("--region", default="auto", help="x,y,ancho,alto, 'auto' (ventana activa) o vacio")
    p.add_argument("--describir", action="store_true")
    p.add_argument("--preguntar", help="pregunta libre sobre la pantalla")
    p.add_argument("--localizar", help="objeto a buscar (madera, arbol, cerdo...)")
    p.add_argument("--ensenar", help="objeto a buscar y ensenar al detector de la IA de juego")
    p.add_argument("--repetir", type=int, default=1)
    p.add_argument("--cada", type=float, default=20, help="segundos entre repeticiones")
    p.add_argument("--espera", type=int, default=3, help="segundos antes de empezar (para ir al juego)")
    p.add_argument("--servidor", default=SERVIDOR)
    p.add_argument("--modelo-vision", default=None, help="id del modelo en LM Studio (por defecto, el de vision)")
    p.add_argument("--salida", default="vision_resultado.png")
    p.add_argument("--dispositivo", default="auto", help=argparse.SUPPRESS)  # la GPU la usa LM Studio
    args = p.parse_args()

    try:
        ia = IAVision(args.servidor, args.modelo_vision)
    except ErrorVision as e:
        raise SystemExit(str(e))
    print(f"IA de vision: {ia.modelo} ({args.servidor})")

    for s in range(args.espera, 0, -1):
        print(f"Empieza en {s}... (pon el juego delante)", flush=True)
        time.sleep(1)
    zona = zona_de(args.region)

    for vez in range(1, args.repetir + 1):
        t0 = time.time()
        rgb = capturar_zona(zona)
        try:
            if args.describir or args.preguntar:
                print("Pensando...", flush=True)
                texto = ia.describir(rgb) if args.describir else ia.preguntar(rgb, args.preguntar)[0]
                print(f"\nLa IA ve:\n{texto}\n", flush=True)
            objeto = args.ensenar or args.localizar
            if objeto:
                print(f"Buscando '{objeto}'...", flush=True)
                cajas, bruto = ia.localizar(rgb, objeto)
                dibujar_cajas(rgb, cajas, args.salida)
                if not cajas:
                    print(f"  No ha encontrado '{objeto}'. Respuesta: {bruto[:200]}", flush=True)
                else:
                    print(f"  Encontrado {len(cajas)} vez/veces: {cajas}  (imagen en {args.salida})", flush=True)
                if args.ensenar and cajas:
                    total = 0
                    for caja in cajas:
                        total = anadir_ejemplo(args.nombre, args.ensenar, recorte_interior(rgb, caja), pantalla=rgb)
                    print(f"  Ensenado '{args.ensenar}' a {args.nombre}: {total} pixeles de ejemplo en total",
                          flush=True)
        except ErrorVision as e:
            raise SystemExit(str(e))
        print(f"  ({time.time() - t0:.1f} s)", flush=True)
        if vez < args.repetir:
            time.sleep(max(0.0, args.cada - (time.time() - t0)))


if __name__ == "__main__":
    main()
