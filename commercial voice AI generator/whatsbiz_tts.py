#!/usr/bin/env python3
"""
whatsbiz_tts.py
-----------------------------------------------------------------------------
Genera la narración en audio del guion de WhatsBiz Pro MedCore usando la
función text-to-speech de Gemini, con una voz masculina comercial.

Requisitos:
    pip install google-genai
    # Opcional, para exportar a MP3:
    pip install pydub   (y tener ffmpeg instalado en el sistema)

Uso:
    export GEMINI_API_KEY="tu_api_key"       # Windows: set GEMINI_API_KEY=...
    python whatsbiz_tts.py

Salida:
    whatsbiz_narracion.wav   (y whatsbiz_narracion.mp3 si pydub está disponible)

Consigue tu API key gratis en: https://aistudio.google.com/apikey
-----------------------------------------------------------------------------
"""

import os
import re
import sys
import time
import wave
import struct

from google import genai
from google.genai import types


from dotenv import load_dotenv
load_dotenv(override=True)   
# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

# Modelo TTS. El modelo Flash SÍ tiene capa gratuita (3 solicitudes/min,
# 15 solicitudes/día). El modelo Pro (gemini-2.5-pro-preview-tts) NO está
# disponible en la capa gratuita; requiere facturación activa.
# OJO: "gemini-3.5-flash" NO sirve aquí: es un modelo de texto, no de voz.
MODEL = "gemini-2.5-flash-preview-tts"

# Voz masculina comercial. Alternativas masculinas recomendadas:
#   Charon      -> Informativa (locutor / comercial)  <-- por defecto
#   Rasalgethi  -> Informativa
#   Orus        -> Firme
#   Alnilam     -> Firme
#   Iapetus     -> Clara
#   Algenib     -> Grave / con textura
#   Sadaltager  -> Con autoridad
#   Puck        -> Juvenil / animada (voz de chico joven)  <-- en uso
#   Fenrir      -> Juvenil y enérgica
VOICE = "Puck"

# Nota de dirección: define el estilo de la locución. Se antepone a cada
# fragmento para mantener un tono consistente en todo el audio.
STYLE_PROMPT = (
    "Lee el siguiente texto en voz alta como un locutor comercial profesional: "
    "voz masculina, cálida y segura, con acento neutro de español mexicano, "
    "ritmo pausado y claro, tono corporativo y confiable, como una voz en off "
    "de un video institucional. No leas estas instrucciones; comienza a narrar "
    "a partir del texto que sigue:\n\n"
)

# Parámetros de audio que Gemini TTS devuelve (PCM crudo).
CHANNELS = 1
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2          # 16-bit

OUTPUT_WAV = "whatsbiz_narracion.wav"
OUTPUT_MP3 = "whatsbiz_narracion.mp3"

MAX_RETRIES = 4           # el modelo TTS a veces devuelve un 500 aleatorio

# --- Ajustes para respetar la capa gratuita -------------------------------
# La capa gratuita permite 3 solicitudes/min y 15/día. Para no pasarnos:
#  - Agrupamos varios párrafos por solicitud (menos solicitudes totales).
#  - Esperamos unos segundos entre solicitudes (< 3 por minuto).
# Si activas facturación (Tier 1), puedes poner PARAGRAPHS_PER_CHUNK = 1 y
# REQUEST_DELAY = 0 para máxima calidad y velocidad.
PARAGRAPHS_PER_CHUNK = 3          # 15 párrafos -> ~5 solicitudes
REQUEST_DELAY = 25                # segundos de espera entre solicitudes

# ---------------------------------------------------------------------------
# EL GUION (español mexicano, profesional)
# ---------------------------------------------------------------------------

SCRIPT = """\
Bienvenido. En este video le mostraremos, paso a paso, cómo instalar, descargar y utilizar la extensión WhatsBiz Pro MedCore, versión 1.0.9, de una manera sencilla y clara.

Abra Google y busque extensiones de Google Chrome. Haga clic en el primer enlace para ingresar a la Chrome Web Store. Dentro de la tienda, escriba WhatsBiz Pro y seleccione la primera tarjeta que aparezca. Haga clic en Añadir a Chrome y confirme con Agregar extensión. Una vez instalada, haga clic en el ícono de la pieza de rompecabezas, en la parte superior derecha del navegador; ubique la extensión y fíjela para tenerla siempre visible.

Ahora abra WhatsApp Web. Si la instalación fue correcta, verá aparecer los elementos de la extensión: un pequeño cohete en la esquina superior derecha, además de nuevas funciones en cada chat y un panel de CRM.

Haga clic en Iniciar sesión y luego en Crear una cuenta. Ingrese su nombre, apellidos, correo electrónico, nombre del negocio —este último es opcional—, contraseña y su confirmación. Acepte los términos y condiciones, y haga clic en Crear cuenta. El sistema le enviará un correo de verificación. Revise su bandeja de entrada y, si no lo encuentra, revise la carpeta de correo no deseado. Abra el correo de WhatsBiz Pro y presione Activar mi cuenta. Si el botón no funciona, copie y pegue el enlace que aparece debajo.

Regrese, ingrese su correo y contraseña, y haga clic en Iniciar sesión. Verá que su cuenta incluye un periodo de prueba gratuito. Si no recibió el correo, vuelva a la página Revisa tu correo y presione Reenviar correo de activación para recibir un nuevo mensaje.

Haga clic en el ícono de la extensión y después en Configuración. Para utilizar la función MedCore, es necesario suscribirse al plan Doctor Pro. Seleccione Actualizar a Pro, realice el pago y regrese a la página de configuración. Desde ahí podrá controlar el tema oscuro o claro, el idioma —español o inglés—, las notificaciones y la personalidad de la inteligencia artificial.

En la sección de personalidad encontrará campos para definir el estilo de comunicación, el contexto del negocio, el estilo de escritura, el idioma, la longitud de las respuestas y el uso de emojis, saludos y despedidas. Si cuenta con un documento PDF —por ejemplo, el manual de su clínica o las políticas para el personal—, puede cargarlo: ingrese el nombre, la descripción y la categoría, seleccione el archivo y presione Subir y analizar documento. La inteligencia artificial leerá el contenido y completará automáticamente los campos de personalidad. Al terminar, presione Guardar personalidad.

Ahora configuraremos el CRM. En WhatsApp, abra un chat y haga clic en el botón verde de la esquina inferior izquierda para abrir el panel. Cambie al CRM de doctor, que contiene los campos diseñados para profesionales de la salud. Antes de acceder al tablero, es necesario completar ciertos datos obligatorios: nombre del establecimiento, Aviso de Funcionamiento ante COFEPRIS, licencia sanitaria, RFC del médico y cédula profesional. Estos son los datos mínimos requeridos. Haga clic en Guardar y obtendrá acceso completo al tablero.

Para mostrar su funcionamiento, regresamos a un chat con un paciente. Si no desea capturar los datos manualmente, presione el botón de inteligencia artificial: en unos segundos, la herramienta extraerá la información de la conversación. Enseguida haga clic en Guardar como paciente y los datos se cargarán a su tablero. Podrá completar la información restante más adelante, ya sea durante la consulta o conforme avance la comunicación por WhatsApp.

En la parte inferior derecha de cada chat encontrará la función de respuestas rápidas, ideal para los mensajes que envía con frecuencia. Por ejemplo, si sus pacientes suelen pedir fotografías de la clínica, puede crear una respuesta con título, descripción e imágenes. Otros ejemplos útiles son los horarios de atención o la ubicación de la clínica, con enlace de Google Maps y Street View. Para enviarla, presione el botón de copiar: la descripción se pegará en el chat y el enlace permitirá ver las imágenes, que el paciente podrá revisar y compartir.

Sobre las respuestas rápidas encontrará el botón Respuesta con IA. Esta función utiliza la personalidad y los documentos que configuró previamente. Si la presiona con el campo de texto vacío, la inteligencia artificial analizará toda la conversación y generará una respuesta acorde con las políticas, el estilo y la profesionalidad de su clínica. Podrá copiarla o generar una nueva. Y si ya escribió algo en el campo de mensaje, la herramienta únicamente lo depurará para que suene más profesional, listo para copiar y enviar.

La función de voz a texto le permite dictar en español o en inglés. Seleccione el idioma, hable, y su voz se convertirá en texto de forma automática.

En la esquina superior derecha verá un pequeño cohete movible. Al hacer clic, se abre un panel flexible donde puede alternar entre tema claro y oscuro, actualizar la información y usar tres botones de acción, además de un campo de texto. Esta herramienta le permite hacer preguntas puntuales sobre el chat en el que se encuentra, sin revisar toda la conversación. Por ejemplo, puede preguntar cuál es el número de este paciente, y gracias al reconocimiento de contexto, la inteligencia artificial le mostrará el número correcto, sin confundirlo con el de un familiar. Al hacer clic en la respuesta, esta se copia al portapapeles. Los tres botones de acción corresponden a las consultas más comunes: precio, dirección y horario. Tenga en cuenta que esta herramienta no es un chat: no recuerda preguntas anteriores; funciona como un buscador de respuestas puntuales, una a la vez.

La última función es la programación de mensajes. Al entrar a un chat, la herramienta lo reconoce. Escriba el mensaje, seleccione la fecha y la hora, y presione Programar mensaje. Verá la lista de mensajes programados, que puede eliminar de forma individual o con Borrar todo. Cuando llegue el momento, aparecerá una ventana emergente con el nombre del paciente, la fecha, la hora y su mensaje. Presione Copiar mensaje y cerrar, abra el chat y péguelo. Es importante señalar que, por las políticas de WhatsApp, el envío no es automático: usted debe pegar y enviar el mensaje. Puede programar recordatorios con días o meses de anticipación; al abrir WhatsApp, estos serán los primeros avisos que verá, incluso si la fecha ya pasó.

Estas son todas las funciones disponibles por ahora. La función de difusión aún se encuentra en fase beta y sigue en desarrollo. Con esto concluye la presentación de WhatsBiz Pro MedCore. Gracias por su atención.
"""

# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------

def split_into_chunks(text: str) -> list[str]:
    """Divide el guion en grupos de párrafos. Gemini TTS pierde calidad en
    clips de varios minutos, pero agrupar unos pocos párrafos reduce el número
    de solicitudes (clave para la capa gratuita) sin degradar el audio."""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    chunks = []
    for i in range(0, len(paragraphs), PARAGRAPHS_PER_CHUNK):
        group = paragraphs[i:i + PARAGRAPHS_PER_CHUNK]
        chunks.append("\n\n".join(group))
    return chunks


def synthesize_chunk(client: genai.Client, text: str) -> bytes:
    """Envía un fragmento a Gemini TTS y devuelve los bytes PCM crudos.
    Reintenta ante los errores 500 esporádicos del modelo."""
    prompt = STYLE_PROMPT + text

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE,
                            )
                        )
                    ),
                ),
            )
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception as e:                       # noqa: BLE001
            if attempt == MAX_RETRIES:
                raise
            wait = 2 * attempt
            print(f"   ⚠ intento {attempt} falló ({e}); reintentando en {wait}s...")
            time.sleep(wait)


def write_wav(filename: str, pcm: bytes) -> None:
    """Envuelve el PCM crudo en un archivo WAV válido."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def try_export_mp3(wav_path: str, mp3_path: str) -> None:
    """Convierte a MP3 si pydub + ffmpeg están disponibles (opcional)."""
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="192k")
        print(f"✔ MP3 exportado: {mp3_path}")
    except Exception:                                # noqa: BLE001
        print("ℹ (Omitido MP3: instala 'pydub' y 'ffmpeg' si lo quieres.)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("❌ Falta la variable de entorno GEMINI_API_KEY.\n"
                 "   Consíguela en https://aistudio.google.com/apikey")

    client = genai.Client(api_key=api_key)

    chunks = split_into_chunks(SCRIPT)
    print(f"Guion dividido en {len(chunks)} fragmentos.")
    print(f"Modelo: {MODEL}  |  Voz: {VOICE}\n")

    all_pcm = bytearray()
    # ~0.35s de silencio entre párrafos para una locución más natural.
    silence = b"\x00\x00" * int(SAMPLE_RATE * 0.35)

    for i, chunk in enumerate(chunks, start=1):
        preview = re.sub(r"\s+", " ", chunk)[:60]
        print(f"[{i}/{len(chunks)}] Generando: {preview}...")
        pcm = synthesize_chunk(client, chunk)
        all_pcm.extend(pcm)
        if i < len(chunks):
            all_pcm.extend(silence)
            # Pausa para no exceder el límite de 3 solicitudes/minuto.
            if REQUEST_DELAY:
                print(f"   ⏳ esperando {REQUEST_DELAY}s (límite de la capa gratuita)...")
                time.sleep(REQUEST_DELAY)

    write_wav(OUTPUT_WAV, bytes(all_pcm))

    duration = len(all_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    print(f"\n✔ WAV guardado: {OUTPUT_WAV}  ({duration/60:.1f} min)")

    try_export_mp3(OUTPUT_WAV, OUTPUT_MP3)
    print("\nListo. 🎙")


if __name__ == "__main__":
    main()