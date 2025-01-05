from fastapi import FastAPI
from routes.detection_routes import router as detection_router

app = FastAPI(
    title="API de Detección Facial",
    description="Procesa detecciones faciales en tiempo real",
    version="1.0"
)

# Registrar rutas
app.include_router(detection_router, prefix="/api/deteccion")

@app.get("/")
async def root():
    return {"mensaje": "API de Detección Facial funcionando correctamente."}
