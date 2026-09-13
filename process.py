from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
from ultralytics import YOLO


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("======================================")
    print("HADARA - PRUEBA GPU + MODELOS")
    print("======================================")

    # ------------------------------------
    # 1. CUDA
    # ------------------------------------

    print("PyTorch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA de PyTorch:", torch.version.cuda)

        device = torch.device("cuda")

    else:

        print("¡¡¡ CUDA NO DISPONIBLE !!!")

        device = torch.device("cpu")


    print("Dispositivo seleccionado:", device)


    # ------------------------------------
    # 2. MI-DAS
    # ------------------------------------

    print("--------------------------------------")
    print("Cargando MiDaS...")

    model = torch.hub.load(
        "intel-isl/MiDaS",
        "DPT_Hybrid",
        pretrained=True
    )

    model.to(device)
    model.eval()

    print("¡¡¡ MiDaS cargado correctamente !!!")


    # ------------------------------------
    # 3. TRANSFORMS
    # ------------------------------------

    print("--------------------------------------")
    print("Cargando transforms...")

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms"
    )

    transform = midas_transforms.dpt_transform

    print("¡¡¡ Transforms cargados correctamente !!!")


    # ------------------------------------
    # 4. YOLO
    # ------------------------------------

    print("--------------------------------------")
    print("Cargando YOLO...")

    yolo = YOLO("yolo11n.pt")

    yolo.to(device)

    print("¡¡¡ YOLO cargado correctamente !!!")


    # ------------------------------------
    # TODO CORRECTO
    # ------------------------------------

    print("======================================")
    print("¡¡¡ TODOS LOS MODELOS CARGADOS !!!")
    print("======================================")

    yield


    print("Servidor apagándose...")


app = FastAPI(lifespan=lifespan)


# ----------------------------------------
# ENDPOINT DE PRUEBA
# ----------------------------------------

@app.get("/ping")
async def ping():

    return {
        "status": "ok",
        "message": "GPU y modelos cargados correctamente"
    }
