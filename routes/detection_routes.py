from fastapi import APIRouter, BackgroundTasks
from services.detection_service import DetectionService

router = APIRouter()
detection_service = DetectionService(video_source="rtsp://admin:Admin123.@192.168.1.100:554/profile2/media.smp")

@router.post("/iniciar")
async def iniciar_deteccion(tareas_fondo: BackgroundTasks):
    """
    Inicia la detección facial en tiempo real.
    """
    detection_service.stop_threads = False
    capture_thread, process_thread = detection_service.start_detection()
    tareas_fondo.add_task(capture_thread.join)
    tareas_fondo.add_task(process_thread.join)
    return {"mensaje": "Detección facial iniciada."}

@router.post("/detener")
async def detener_deteccion():
    """
    Detiene la detección facial en tiempo real.
    """
    detection_service.stop_detection()
    return {"mensaje": "Detección facial detenida."}

@router.get("/resumen")
async def obtener_resumen():
    """
    Genera un resumen basado en las detecciones de los últimos 30 segundos.
    """
    resumen = detection_service.get_summary()
    return resumen