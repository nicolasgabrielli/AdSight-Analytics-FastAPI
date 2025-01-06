from fastapi import APIRouter, BackgroundTasks
from services.detection_service import DetectionService

router = APIRouter()

# Inicializamos el servicio pero no iniciamos la detección
detection_service = DetectionService(
    video_source="rtsp://admin:Admin123.@192.168.1.100:554/profile2/media.smp",
    high_resolution=True
)
detection_service.stop_threads = True  # Aseguramos que inicie detenido


@router.post("/iniciar")
async def iniciar_deteccion(tareas_fondo: BackgroundTasks):
    """
    Inicia la detección facial en tiempo real.
    """
    # Verifica si la detección ya está en ejecución
    if detection_service.stop_threads is False:
        return {"mensaje": "La detección facial ya está en ejecución."}

    detection_service.stop_threads = False

    # Inicia las tareas de detección en segundo plano
    capture_thread, process_thread, summary_thread = detection_service.start_detection()

    # Agrega las tareas al manejador de BackgroundTasks para que se ejecuten de forma segura
    tareas_fondo.add_task(capture_thread.join)
    tareas_fondo.add_task(process_thread.join)
    tareas_fondo.add_task(summary_thread.join)

    return {"mensaje": "Detección facial iniciada."}


@router.post("/detener")
async def detener_deteccion():
    """
    Detiene la detección facial en tiempo real.
    """
    if detection_service.stop_threads:
        return {"mensaje": "La detección facial ya está detenida."}

    detection_service.stop_detection()
    return {"mensaje": "Detección facial detenida."}


@router.get("/resumen")
async def obtener_resumen():
    """
    Genera un resumen basado en las detecciones de los últimos 30 segundos.
    """
    resumen = detection_service.get_summary()
    return resumen
