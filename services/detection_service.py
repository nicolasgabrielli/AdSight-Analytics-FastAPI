import cv2
from deepface import DeepFace
import tensorflow as tf
from threading import Thread, Lock
from queue import Queue
import time
import uuid
import math
import json
import requests

# Configuración de GPU
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detectada y configurada: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"Error al configurar la GPU: {e}")
    else:
        print("No se detectaron GPUs, utilizando CPU.")

setup_gpu()

# URL del backend de Spring Boot fija
backend_url = "http://localhost:8080/api/resumen"


class DetectionService:
    def __init__(self, video_source=0):
        self.stop_threads = False
        self.video_source = video_source
        self.persons = {}
        self.detections_last_30_seconds = []  # Lista para almacenar detecciones de los últimos 30 segundos
        self.lock = Lock()
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.backend_url = backend_url

    def calculate_distance(self, coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def assign_persistent_id(self, region, analysis, max_distance=100):
        """
        Asigna un ID persistente basado en la proximidad de coordenadas detectadas y actualiza la información de la persona.
        """
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
        center_new = (x + w // 2, y + h // 2)
        current_time = time.time()

        with self.lock:
            for person_id, data in self.persons.items():
                center_existing = data["center"]
                if self.calculate_distance(center_existing, center_new) < max_distance:
                    # Actualizar tiempo de visualización
                    time_in_view = current_time - data["last_seen"]
                    data["time_in_screen"] += time_in_view
                    data["last_seen"] = current_time

                    # Actualizar otros datos detectados
                    data["center"] = center_new
                    data["age"].append(analysis.get("age", 0))
                    data["gender"][analysis.get("dominant_gender", "N/A")] += 1
                    data["race"][analysis.get("dominant_race", "N/A")] += 1
                    data["emotion"][analysis.get("dominant_emotion", "N/A")] += 1
                    return person_id

            # Crear un nuevo ID si no se encuentra un cercano
            new_id = str(uuid.uuid4())
            self.persons[new_id] = {
                "center": center_new,
                "time_in_screen": 0,
                "last_seen": current_time,
                "age": [analysis.get("age", 0)],
                "gender": {"Man": 0, "Woman": 0},
                "race": {},
                "emotion": {},
            }
            self.persons[new_id]["gender"][analysis.get("dominant_gender", "N/A")] += 1
            self.persons[new_id]["race"][analysis.get("dominant_race", "N/A")] = 1
            self.persons[new_id]["emotion"][analysis.get("dominant_emotion", "N/A")] = 1
            return new_id


    def capture_frames(self):
        vid = cv2.VideoCapture(self.video_source)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while not self.stop_threads:
            ret, frame = vid.read()
            if not ret:
                print("No se pudo capturar el frame.")
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                print("Frame capturado.")

        vid.release()
        print("Captura de frames detenida.")

    def process_frames(self):
        while not self.stop_threads:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                print("Procesando frame...")
                resized_frame = cv2.resize(frame, (640, 360))
                try:
                    results = DeepFace.analyze(
                        resized_frame,
                        actions=['age', 'gender', 'race', 'emotion'],
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    for analysis in results if isinstance(results, list) else [results]:
                        region = analysis.get("region", {})
                        self.assign_persistent_id(region, analysis)
                    print("Resultados procesados:", results)
                except Exception as e:
                    print(f"Error al analizar el rostro: {e}")

        print("Procesamiento de frames detenido.")



    def get_summary(self):
        """
        Genera un resumen basado en los datos recopilados en los últimos 30 segundos.
        Incluye el tiempo acumulado de visualización para cada persona.
        """
        summary = {
            "total_personas": 0,
            "personas": []
        }

        current_time = time.time()

        with self.lock:
            for person_id, data in self.persons.items():
                # Filtrar personas vistas en los últimos 30 segundos
                if current_time - data["last_seen"] <= 30:
                    summary["total_personas"] += 1

                    # Calcular el promedio de edad
                    avg_age = sum(data["age"]) / len(data["age"]) if data["age"] else 0

                    # Determinar género, raza y emoción predominantes
                    dominant_gender = max(data["gender"], key=data["gender"].get)
                    dominant_race = max(data["race"], key=data["race"].get)
                    dominant_emotion = max(data["emotion"], key=data["emotion"].get)

                    summary["personas"].append({
                        "id": person_id[:8],
                        "edad_promedio": round(avg_age, 2),
                        "genero_predominante": dominant_gender,
                        "raza_predominante": dominant_race,
                        "emocion_predominante": dominant_emotion,
                        "tiempo_en_pantalla": round(data["time_in_screen"], 2)
                    })

        return summary

    
    def send_summary_to_backend(self, backend_url):
        """
        Envia el resumen actual al backend en formato JSON.
        """
        while not self.stop_threads:
            time.sleep(30)  # Esperar 30 segundos
            summary = self.get_summary()  # Obtener el resumen actual

            try:
                # Convertir el resumen a JSON
                json_data = json.dumps(summary)

                # Enviar el resumen al backend
                response = requests.post(
                    backend_url,
                    data=json_data,
                    headers={"Content-Type": "application/json"}
                )

                # Log para verificar que se envió correctamente
                if response.status_code == 200:
                    print(f"Resumen enviado exitosamente al backend: {response.json()}")
                else:
                    print(f"Error al enviar el resumen al backend: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Error al intentar enviar el resumen: {e}")

    def start_detection(self):
        """
        Inicia la captura, procesamiento y envío de resúmenes con una URL fija.
        """
        self.stop_threads = False
        capture_thread = Thread(target=self.capture_frames, daemon=True)
        process_thread = Thread(target=self.process_frames, daemon=True)
        summary_thread = Thread(target=self.send_summary_to_backend, args=(backend_url,), daemon=True)

        capture_thread.start()
        process_thread.start()
        summary_thread.start()
        return capture_thread, process_thread


    def start_detection(self):
        self.stop_threads = False
        capture_thread = Thread(target=self.capture_frames, daemon=True)
        process_thread = Thread(target=self.process_frames, daemon=True)
        capture_thread.start()
        process_thread.start()
        return capture_thread, process_thread

    def stop_detection(self):
        print("Deteniendo detección...")
        self.stop_threads = True
