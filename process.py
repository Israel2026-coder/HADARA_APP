from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("===================================")
    print("HADARA - SERVIDOR DE PRUEBA")
    print("Iniciando servidor...")
    print("===================================")

    yield

    print("Servidor apagándose...")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(lifespan=lifespan)


# ============================================================
# CORS
# ============================================================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# TEST 1 — PING
# ============================================================

@app.get("/ping")
async def ping():

    return {
        "status": "ok",
        "message": "HADARA está funcionando"
    }


# ============================================================
# TEST 2 — INFORMACIÓN DEL SERVIDOR
# ============================================================

@app.get("/")
async def root():

    return {
        "server": "HADARA",
        "status": "online",
        "message": "Servidor funcionando correctamente"
    }


# ============================================================
# TEST 3 — RECIBIR TEXTO
# ============================================================

@app.post("/test")
async def test(message: str = Form(...)):

    print("Mensaje recibido:", message)

    return {
        "status": "ok",
        "received": message
    }


# ============================================================
# TEST 4 — RECIBIR IMAGEN
# ============================================================

@app.post("/process")
async def process(
    image: UploadFile = File(...),
    distance: float = Form(...)
):

    print("===================================")
    print("PETICIÓN RECIBIDA")
    print("===================================")

    print("Nombre:", image.filename)
    print("Tipo:", image.content_type)
    print("Distancia:", distance)

    # Leer bytes de la imagen
    contents = await image.read()

    print("Bytes recibidos:", len(contents))

    # Intentar abrir la imagen con PIL
    try:

        img = Image.open(
            BytesIO(contents)
        )

        print("Formato detectado:", img.format)
        print("Resolución:", img.size)
        print("Modo:", img.mode)

        # Convertir a RGB exactamente como hará
        # tu programa real
        img = img.convert("RGB")

        print("Conversión RGB: OK")

    except Exception as e:

        print("ERROR PROCESANDO IMAGEN:", e)

        return {
            "status": "error",
            "message": "La imagen no pudo ser interpretada",
            "error": str(e)
        }


    # ========================================================
    # RESPUESTA
    # ========================================================

    return {

        "status": "ok",

        "message": "Imagen recibida correctamente",

        "filename": image.filename,

        "content_type": image.content_type,

        "bytes": len(contents),

        "format": img.format,

        "width": img.width,

        "height": img.height,

        "distance": distance

    }
