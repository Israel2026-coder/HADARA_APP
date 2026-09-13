from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO



# Variables globales
model = None
transform = None
device = None
yolo = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform, device, yolo

    print("Iniciando servidor...")

    # Se ejecuta UNA sola vez al arrancar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando:", device)

    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    yolo =  YOLO("yolo11n.pt")
    yolo.to(device)
    print("Modelo cargado.")

    yield

    # opcional al apagar
    print("Apagando servidor...")

#declararemos todos los métodos que teníamos antes
def pil_to_yolo_input(pil_image):
    # PIL (RGB) → numpy
    img = np.array(pil_image)

    # RGB → BGR (como cv2.imread)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img
def detect_objects_yolo(image_pil, model, conf_threshold=0.47):
    img = pil_to_yolo_input(image_pil)
    results = model(img)
    r = results[0]

    detections = []

    if r.boxes is None:
        return detections, r.orig_shape[1]

    for box in r.boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        class_name = r.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": class_name,
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence
        })

    image_width = r.orig_shape[1]
    return detections, image_width

def get_objects(image_pil, yolo):
    detections, image_width = detect_objects_yolo(image_pil, yolo)

    objects = []

    # Objetos detectados por YOLO
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        objects.append({
            "label": det["class"],
            "box": (x1, y1, x2, y2)
        })

    # ---------------------------------
    # Agregar box centrado de 200x200
    # ---------------------------------
    img_width, img_height = image_pil.size

    box_size = 200
    half = box_size // 2

    center_x = img_width // 2
    center_y = img_height // 2

    x1 = center_x - half
    y1 = center_y - half
    x2 = center_x + half
    y2 = center_y + half

    objects.append({
        "label": "center",
        "box": (x1, y1, x2, y2)
    })

    return objects

def process_objects(image_pil, depth_map, distance, yolo ):
    objects = get_objects(image_pil, yolo)

    temp_results = []
    center_depth = None

    # -----------------------------------------
    # Primero calcular profundidades relativas
    # -----------------------------------------
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]

        if x2 <= x1 or y2 <= y1:
            continue

        region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            continue

        # primer intento: depth_value = np.percentile(region, 30)
        depth_value = np.median(region)
        box_size = ((x2 - x1) + (y2 - y1)) / 2

        item = {
            "label": obj["label"],
            "depth": float(depth_value),
            "size": float(box_size),
            "box": (x1, y1, x2, y2)
        }

        temp_results.append(item)

        # Guardar profundidad del cuadro central
        if obj["label"] == "center":
            center_depth = float(depth_value)

    # Si no existe center o depth inválido
    if center_depth is None or center_depth == 0:
        return []

    # -----------------------------------------
    # Convertir profundidad relativa a distancia real
    # regla proporcional
    # -----------------------------------------
    results = []

    for item in temp_results:
        k = 1.25
        real_distance = distance * (center_depth / item["depth"]) ** k
        results.append({
            "label": item["label"],
            "distance": float(real_distance),
            "size": item["size"],
            "box": item["box"]
        })

    return results

def get_depth_map(image_pil, model, transform):
    img = np.array(image_pil)  # PIL → numpy (RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map
#path = "/content/prueba_distance.jpeg"

"""Adaptaré el parámetro "path" para que reciba la imagen ya procesada.
En vez de (path, distance) ahora será (image_pil, distance)""" 
def get_second_gender_observation(image_pil, distance, yolo, model, transform):

  # cargar imagen UNA sola vez
  #image_pil = Image.open(path).convert("RGB"), Esto ya no es necesario

  # 1. profundidad
  depth_map = get_depth_map(image_pil, model, transform)

  # 2. objetos + depth

  objects_info = process_objects(image_pil, depth_map, distance, yolo)

  # 3. ver resultados
  for obj in objects_info:
      print(obj)
  return objects_info
def results_to_prompt(results):
    return "\n".join(
        f"There is {'an' if str(r['label'])[0].lower() in 'aeiou' else 'a'} {r['label']} "
        f"{r['distance']:.2f} meters away at coordinates {r['box']}. "
        f"Its id is {i}"
        for i, r in enumerate(results)
    )
    
app = FastAPI(lifespan=lifespan)

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    distance: float = Form(...)
):
    #global model, transform, device

    # aquí ya existen y están cargados
    contents = await image.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    """En estqe código con respecto al GetSecondObservation original, no necesito un path físico 
    para obtener el mismo resultado que Image.open(path).convert("RGB"). Lo que recibo (contents = await file.read())
     son bytes de la imagen, y PIL.Image.open() también puede abrir desde memoria usando BytesIO."""

    # luego procesas imagen...

    results = get_second_gender_observation(
    img,
    distance,
    yolo,
    model,
    transform
)
    prompt = results_to_prompt(results)
    return {"respuesta": prompt}
