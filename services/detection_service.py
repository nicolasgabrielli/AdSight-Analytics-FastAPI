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
        self.frame_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.persons = {}
        self.analysis_frequency = 1
        self.last_result = None
        self.high_resolution = True
        
        # Ajustar umbrales
        self.confidence_thresholds = {
            'gender': 0.65,
            'race': 0.60,
            'emotion': 0.40,
            'age': 0.65
        }
        
        self.temporal_window = 2
        self.detection_history = {}
        
        # Mapeo de emociones
        self.emotion_mapping = {
            'happy': 'FELIZ',
            'sad': 'TRISTE',
            'angry': 'ENOJADO',
            'fear': 'ASUSTADO',
            'surprise': 'SORPRENDIDO',
            'neutral': 'NEUTRAL',
            'disgust': 'DISGUSTADO',
            'contempt': 'DESPRECIO'
        }

    def calculate_distance(self, coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def assign_persistent_id(self, region, analysis, max_distance=100):
        try:
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            center_new = (x + w // 2, y + h // 2)
            current_time = time.time()

            with self.lock:
                for person_id, data in self.persons.items():
                    center_existing = data["center"]
                    time_since_last_seen = current_time - data["last_seen"]
                    
                    if time_since_last_seen > 1:  # Reducido a 1 segundo
                        continue
                    
                    if self.calculate_distance(center_existing, center_new) < max_distance:
                        time_diff = current_time - data["last_seen"]
                        data["time_in_screen"] += time_diff
                        data["last_seen"] = current_time
                        data["center"] = center_new
                        
                        # Mantener solo las últimas 2 detecciones para edad
                        data["age"].append(analysis.get("age", 0))
                        if len(data["age"]) > 2:
                            data["age"].pop(0)
                        
                        # Actualizar emociones inmediatamente
                        emotion = analysis.get("dominant_emotion", "N/A")
                        data["emotion"] = {emotion: 1}  # Reset y actualizar directamente
                        
                        # Actualizar otros atributos normalmente
                        gender = analysis.get("dominant_gender", "N/A")
                        race = analysis.get("dominant_race", "N/A")
                        
                        data["gender"][gender] = data["gender"].get(gender, 0) + 1
                        data["race"][race] = data["race"].get(race, 0) + 1
                        
                        return person_id

                # Limpiar personas que no se han visto en más de 1 segundo
                expired_ids = [pid for pid, data in self.persons.items() 
                             if current_time - data["last_seen"] > 1]
                for expired_id in expired_ids:
                    del self.persons[expired_id]

                # Crear nueva persona con contadores inicializados
                new_id = str(uuid.uuid4())
                self.persons[new_id] = {
                    "center": center_new,
                    "time_in_screen": 0,
                    "last_seen": current_time,
                    "age": [analysis.get("age", 0)],
                    "gender": {analysis.get("dominant_gender", "N/A"): 1},
                    "race": {analysis.get("dominant_race", "N/A"): 1},
                    "emotion": {analysis.get("dominant_emotion", "N/A"): 1}
                }
                return new_id

        except Exception as e:
            print(f"Error al asignar ID: {str(e)}")
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

    def validate_detection(self, analysis):
        """Validar la confianza de la detección"""
        if not analysis.get('confidence', 0) > self.confidence_thresholds['gender']:
            return False
            
        return True

    def smooth_predictions(self, person_id, current_prediction):
        """Suavizar predicciones usando una ventana temporal"""
        if person_id not in self.detection_history:
            self.detection_history[person_id] = []
            
        history = self.detection_history[person_id]
        history.append(current_prediction)
        
        # Mantener solo los últimos N frames
        if len(history) > self.temporal_window:
            history.pop(0)
            
        # Promediar predicciones
        smoothed = {
            'age': sum(h['age'] for h in history) / len(history),
            'gender': max(set(h['gender'] for h in history), key=history.count),
            'race': max(set(h['race'] for h in history), key=history.count),
            'emotion': max(set(h['emotion'] for h in history), key=history.count)
        }
        
        return smoothed

    def process_frames_hd(self):
        while not self.stop_threads:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                try:
                    # Reducir resolución para mejor rendimiento pero mantener calidad
                    processed_frame = cv2.resize(frame, (640, 360))  # 640p
                    
                    # Mejorar la calidad de la imagen
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    results = DeepFace.analyze(
                        processed_frame,
                        actions=['age', 'gender', 'race', 'emotion'],
                        detector_backend="retinaface",
                        enforce_detection=False,
                        align=True,
                        silent=True
                    )
                    
                    if isinstance(results, dict):
                        results = [results]

                    valid_results = []
                    for analysis in results:
                        region = analysis.get("region", {})
                        if region:
                            # Ajustar el escalado para 640p
                            scale_x = frame.shape[1] / 640
                            scale_y = frame.shape[0] / 360
                            
                            region['x'] = int(region['x'] * scale_x)
                            region['y'] = int(region['y'] * scale_y)
                            region['w'] = int(region['w'] * scale_x)
                            region['h'] = int(region['h'] * scale_y)
                            
                            face_area = region['w'] * region['h']
                            frame_area = frame.shape[0] * frame.shape[1]
                            
                            if 0.001 <= (face_area / frame_area) <= 0.5:
                                person_id = self.assign_persistent_id(region, analysis)
                                if person_id:
                                    valid_results.append(analysis)

                    self.last_result = valid_results
                    
                except Exception as e:
                    print(f"Error en análisis: {str(e)}")

    def process_frames_sd(self):
        while not self.stop_threads:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                resized_frame = cv2.resize(frame, (240, 135))
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
                    
                    # Ajustar el centro para enfocarse en la cara
                    center_x = x + w // 2
                    center_y = y + h // 2  # Volver al centro real
                    
                    # Hacer el recuadro cuadrado y más grande
                    size = int(max(w, h) * 1.2)  # Aumentar 20%
                    
                    # Ajustar posición para centrar en la cara
                    x = max(0, center_x - size // 2)
                    y = max(0, center_y - size // 2)
                    
                    # Asegurar que el recuadro no se salga del frame
                    w = min(frame.shape[1] - x, size)
                    h = min(frame.shape[0] - y, size)

                    # Dibujar rectángulo con borde doble
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 5)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

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

                        info_text = [
                            f"ID: {person_id[:8]}",
                            f"Tiempo: {round(person_data['time_in_screen'], 1)}s",
                            f"Edad: {int(avg_age)}",
                            f"Genero: {dominant_gender}",
                            f"Raza: {dominant_race}",
                            f"Emocion: {dominant_emotion}"
                        ]

                        # Posicionar el texto al lado del rectángulo
                        text_x = x + w + 10
                        text_y = y

                        # Dibujar fondo semi-transparente para el texto
                        overlay = frame.copy()
                        text_padding = 20
                        text_height = len(info_text) * 35
                        cv2.rectangle(
                            overlay,
                            (text_x - text_padding, text_y - text_padding),
                            (text_x + 300, text_y + text_height),
                            (0, 0, 0),
                            -1
                        )
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                        # Dibujar texto con borde para mejor legibilidad
                        font_scale = 0.8
                        font_thickness = 2
                        for i, text in enumerate(info_text):
                            text_pos_y = text_y + (i * 35) + 25
                            # Borde negro
                            cv2.putText(
                                frame, text, (text_x, text_pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 
                                font_thickness + 2, cv2.LINE_AA
                            )
                            # Texto blanco
                            cv2.putText(
                                frame, text, (text_x, text_pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                                font_thickness, cv2.LINE_AA
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
