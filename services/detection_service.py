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
from datetime import datetime

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

# URL del backend de Spring Boot
backend_url = "http://localhost:8080/api/resumen"
id_pantalla = "1"

class DetectionService:
    def __init__(self, video_source=0, high_resolution=True):
        self.stop_threads = False
        self.video_source = video_source
        self.frame_queue = Queue(maxsize=4)
        self.lock = Lock()
        self.persons = {}
        self.analysis_frequency = 4
        self.last_result = None
        self.high_resolution = high_resolution

    def calculate_distance(self, coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def assign_persistent_id(self, region, analysis, max_distance=100):
        try:
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            center_new = (x + w // 2, y + h // 2)
            current_time = time.time()

            with self.lock:
                # Ignorar análisis que no contengan los datos necesarios
                if not all(key in analysis for key in ['age', 'dominant_gender', 'dominant_race', 'dominant_emotion']):
                    return None

                # Buscar una persona existente que coincida con la posición
                for person_id, data in self.persons.items():
                    center_existing = data["center"]
                    time_since_last_seen = current_time - data["last_seen"]
                    
                    # Si han pasado más de 5 segundos, ignorar esta persona
                    if time_since_last_seen > 5:
                        continue
                    
                    if self.calculate_distance(center_existing, center_new) < max_distance:
                        # Actualizar datos de la persona existente
                        time_diff = current_time - data["last_seen"]
                        data["time_in_screen"] += time_diff
                        data["last_seen"] = current_time
                        data["center"] = center_new
                        data["age"].append(analysis.get("age", 0))
                        data["gender"][analysis.get("dominant_gender", "N/A")] += 1
                        data["race"][analysis.get("dominant_race", "N/A")] = data["race"].get(analysis.get("dominant_race", "N/A"), 0) + 1
                        data["emotion"][analysis.get("dominant_emotion", "N/A")] = data["emotion"].get(analysis.get("dominant_emotion", "N/A"), 0) + 1
                        return person_id

                # Limpiar personas que no se han visto en más de 5 segundos
                expired_ids = [pid for pid, data in self.persons.items() 
                             if current_time - data["last_seen"] > 5]
                for expired_id in expired_ids:
                    del self.persons[expired_id]

                # Crear nueva persona
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
            
        except Exception as e:
            error_str = str(e)
            if not any(val in error_str.lower() for val in ['neutral', 'white', 'fear', 'sad']):
                print(f"Error al asignar ID: {error_str}")
            return None

    def capture_frames(self):
        vid = cv2.VideoCapture(self.video_source)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        vid.set(cv2.CAP_PROP_FPS, 30)

        cv2.namedWindow("Captura de frames", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Captura de frames", 1280, 720)

        frame_count = 0

        while not self.stop_threads:
            ret, frame = vid.read()
            if not ret:
                print("No se pudo capturar el fotograma.")
                break

            frame_count += 1

            if frame_count % self.analysis_frequency == 0 and not self.frame_queue.full():
                self.frame_queue.put(frame)

            processed_frame = self.draw_analysis_data(frame)
            cv2.imshow("Captura de frames", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_threads = True
                break

        vid.release()
        cv2.destroyAllWindows()

    def process_frames_hd(self):
        while not self.stop_threads:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                try:
                    # Analizar directamente en 720p
                    results = DeepFace.analyze(
                        frame,
                        actions=['age', 'gender', 'race', 'emotion'],
                        detector_backend="opencv",
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(results, dict):
                        results = [results]
                    elif not isinstance(results, list):
                        results = []

                    # Procesar cada detección
                    valid_results = []
                    for analysis in results:
                        region = analysis.get("region", {})
                        # Validar que la región sea válida y no cubra más del 50% del frame
                        if (region and 
                            isinstance(region, dict) and 
                            all(k in region for k in ['x', 'y', 'w', 'h'])):
                            
                            # Calcular el área del rostro y del frame
                            frame_area = frame.shape[0] * frame.shape[1]
                            face_area = region['w'] * region['h']
                            
                            # Si el área del rostro es menor al 30% del frame, es probablemente válida
                            if face_area < (frame_area * 0.3):
                                try:
                                    person_id = self.assign_persistent_id(region, analysis)
                                    if person_id:
                                        valid_results.append(analysis)
                                except Exception as e:
                                    if not str(e).lower() in ['neutral', 'white', 'fear', 'sad']:
                                        print(f"Error al asignar ID: {str(e)}")
                    
                    self.last_result = valid_results
                    
                except Exception as e:
                    error_str = str(e)
                    if not any(val in error_str.lower() for val in ['neutral', 'white', 'fear', 'sad']):
                        print(f"Error en análisis: {error_str}")

    def process_frames_sd(self):
        while not self.stop_threads:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                resized_frame = cv2.resize(frame, (320, 180))
                try:
                    results = DeepFace.analyze(
                        resized_frame,
                        actions=['age', 'gender', 'race', 'emotion'],
                        detector_backend="opencv",
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(results, dict):
                        results = [results]
                    elif not isinstance(results, list):
                        results = []

                    # Ajustar las coordenadas a la resolución original (720p)
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 180

                    # Procesar cada detección
                    valid_results = []
                    for result in results:
                        if 'region' in result:
                            region = result['region']
                            # Escalar las coordenadas
                            region['x'] = int(region['x'] * scale_x)
                            region['y'] = int(region['y'] * scale_y)
                            region['w'] = int(region['w'] * scale_x)
                            region['h'] = int(region['h'] * scale_y)

                            # Validar que la región sea válida y no cubra más del 30% del frame
                            frame_area = frame.shape[0] * frame.shape[1]
                            face_area = region['w'] * region['h']
                            
                            if face_area < (frame_area * 0.3):
                                try:
                                    person_id = self.assign_persistent_id(region, result)
                                    if person_id:
                                        valid_results.append(result)
                                except Exception as e:
                                    if not str(e).lower() in ['neutral', 'white', 'fear', 'sad']:
                                        print(f"Error al asignar ID: {str(e)}")
                    
                    self.last_result = valid_results
                    
                except Exception as e:
                    error_str = str(e)
                    if not any(val in error_str.lower() for val in ['neutral', 'white', 'fear', 'sad']):
                        print(f"Error en análisis: {error_str}")

    def process_frames(self):
        if self.high_resolution:
            return self.process_frames_hd()
        else:
            return self.process_frames_sd()

    def draw_analysis_data(self, frame):
        try:
            if self.last_result and len(self.last_result) > 0:
                for analysis in self.last_result:
                    region = analysis.get("region", {})
                    if not region:
                        continue

                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                    # Dibujar rectángulo alrededor del rostro
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Encontrar el ID persistente para esta detección
                    center_new = (x + w // 2, y + h // 2)
                    person_data = None
                    person_id = None

                    with self.lock:
                        for pid, data in self.persons.items():
                            if self.calculate_distance(data["center"], center_new) < 100:
                                person_id = pid
                                person_data = data
                                break

                    if person_data:
                        # Calcular datos promedio
                        avg_age = sum(person_data["age"]) / len(person_data["age"])
                        dominant_gender = max(person_data["gender"], key=person_data["gender"].get)
                        dominant_race = max(person_data["race"], key=person_data["race"].get) if person_data["race"] else "N/A"
                        dominant_emotion = max(person_data["emotion"], key=person_data["emotion"].get) if person_data["emotion"] else "N/A"

                        # Preparar texto con toda la información
                        info_text = [
                            f"ID: {person_id[:8]}",
                            f"Tiempo: {round(person_data['time_in_screen'], 1)}s",
                            f"Edad: {int(avg_age)}",
                            f"Genero: {dominant_gender}",
                            f"Raza: {dominant_race}",
                            f"Emocion: {dominant_emotion}"
                        ]

                        # Ajustar tamaño de fuente para 720p
                        font_scale = 1.0  # Aumentado para mejor visibilidad en 720p
                        font_thickness = 2
                        padding = 10

                        # Posicionar el texto encima del rectángulo
                        text_y = y - padding - (len(info_text) * 30)  # Aumentado el espaciado
                        if text_y < 0:  # Si está muy arriba, ponerlo debajo
                            text_y = y + h + 30

                        # Dibujar cada línea de texto con su fondo
                        for i, text in enumerate(info_text):
                            text_pos_y = text_y + (i * 30)  # Aumentado el espaciado vertical
                            
                            # Obtener el tamaño del texto
                            (text_width, text_height), _ = cv2.getTextSize(
                                text, 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                font_scale, 
                                font_thickness
                            )
                            
                            # Dibujar rectángulo de fondo con padding
                            cv2.rectangle(
                                frame, 
                                (x - padding, text_pos_y - text_height - padding),
                                (x + text_width + padding, text_pos_y + padding),
                                (0, 0, 0), 
                                -1
                            )
                            
                            # Dibujar texto
                            cv2.putText(
                                frame, 
                                text,
                                (x, text_pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, 
                                (255, 255, 255), 
                                font_thickness,
                                cv2.LINE_AA
                            )

        except Exception as e:
            print(f"Error al dibujar datos: {e}")

        return frame

    def send_summary_to_backend(self, backend_url="http://localhost:8080/api/resumen"):
        while not self.stop_threads:
            time.sleep(30)  # Espera 30 segundos
            summary = self.get_summary()

            if summary["total_personas"] > 0:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwiaWF0IjoxNzEwMjY5NjI0LCJleHAiOjE3MTI4NjE2MjR9.YOl_PQtUhGDHUOmBub_dq_8EQtXOPKYtAFQtEXhXAYE"
                    }

                    # Ajustando el formato según la respuesta del backend
                    json_data = {
                        "total_personas": summary["total_personas"],  # usando snake_case como en la respuesta
                        "id_pantalla": "1",
                        "fecha": datetime.now().isoformat(),  # agregando fecha
                        "personas": [
                            {
                                "id": persona["id"],
                                "edad_promedio": persona["edad_promedio"],
                                "genero_predominante": persona["genero_predominante"],
                                "raza_predominante": persona["raza_predominante"],
                                "emocion_predominante": persona["emocion_predominante"],
                                "tiempo_en_pantalla": persona["tiempo_en_pantalla"]
                            }
                            for persona in summary["personas"]
                        ]
                    }

                    print(f"Enviando datos: {json_data}")  # Debug

                    response = requests.post(
                        backend_url,
                        json=json_data,
                        headers=headers
                    )

                    if response.status_code == 200:
                        print(f"Resumen enviado exitosamente: {response.json()}")
                    elif response.status_code == 401:
                        print("Error de autenticación: Token expirado o inválido")
                    else:
                        print(f"Error al enviar el resumen: {response.status_code}, {response.text}")
                except Exception as e:
                    print(f"Error al intentar enviar el resumen: {e}")

    def get_summary(self):
        summary = {
            "total_personas": 0,
            "personas": []
        }

        current_time = time.time()
        expired_ids = []

        with self.lock:
            # Identificar y eliminar personas que no se han visto en más de 30 segundos
            for person_id, data in self.persons.items():
                if current_time - data["last_seen"] > 30:
                    expired_ids.append(person_id)
                # Solo incluir personas con más de 1 segundo en pantalla
                elif data["time_in_screen"] >= 1.0:
                    summary["total_personas"] += 1
                    avg_age = sum(data["age"]) / len(data["age"]) if data["age"] else 0
                    dominant_gender = max(data["gender"], key=data["gender"].get)
                    dominant_race = max(data["race"], key=data["race"].get)
                    dominant_emotion = max(data["emotion"], key=data["emotion"].get)

                    summary["personas"].append({
                        "id": person_id[:8],
                        "edad_promedio": round(avg_age, 2),
                        "genero_predominante": dominant_gender,
                        "raza_predominante": dominant_race,
                        "emocion_predominante": dominant_emotion,
                        "tiempo_en_pantalla": round(data["time_in_screen"], 2),
                        "id_pantalla": id_pantalla
                    })

            # Eliminar personas expiradas
            for expired_id in expired_ids:
                del self.persons[expired_id]

        return summary

    def start_detection(self):
        self.stop_threads = False
        capture_thread = Thread(target=self.capture_frames, daemon=True)
        process_thread = Thread(target=self.process_frames, daemon=True)
        summary_thread = Thread(target=self.send_summary_to_backend, args=(backend_url,), daemon=True)

        capture_thread.start()
        process_thread.start()
        summary_thread.start()

        return capture_thread, process_thread, summary_thread

    def stop_detection(self):
        self.stop_threads = True
        print("Detección detenida.")
