from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

# 🔹 Endpoint de ping (para tu función pingServer)
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# 🔹 Endpoint principal (imagen + distancia)
@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    distance: float = Form(...)
):
    # Leer imagen (por ahora no la usamos)
    contents = await file.read()

    print(f"Imagen recibida: {file.filename}")
    print(f"Distancia: {distance}")

    # Aquí luego meterás ML

    return JSONResponse(content={"message": "copiado"})
