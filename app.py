from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import asyncio
import threading
from telegram import Bot
from datetime import datetime
import logging

app = Flask(__name__)

# Configuración
TELEGRAM_BOT_TOKEN = os.environ.get('8089007271:AAEUKn7JOx56JjREetduUsq8Qw3PFRewuD8')
TELEGRAM_CHAT_ID = os.environ.get('-4893597683')
MODEL_PATH = 'models/impresion.pt'

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorDetector:
    def __init__(self):
        self.model = None
        self.bot = None
        self.load_model()
        self.setup_telegram()
    
    def load_model(self):
        """Cargar modelo YOLOv5"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=MODEL_PATH, force_reload=True)
            self.model.eval()
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
    
    def setup_telegram(self):
        """Configurar bot de Telegram"""
        if TELEGRAM_BOT_TOKEN:
            self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
            logger.info("Bot de Telegram configurado")
    
    def detect_errors(self, image):
        """Detectar errores en la imagen"""
        try:
            results = self.model(image)
            detections = results.pandas().xyxy[0]
            
            errors_found = []
            for _, detection in detections.iterrows():
                if detection['confidence'] > 0.5:  # Umbral de confianza
                    errors_found.append({
                        'class': detection['name'],
                        'confidence': float(detection['confidence']),
                        'bbox': [
                            float(detection['xmin']),
                            float(detection['ymin']),
                            float(detection['xmax']),
                            float(detection['ymax'])
                        ]
                    })
            
            return errors_found
        except Exception as e:
            logger.error(f"Error en detección: {e}")
            return []
    
    async def send_telegram_alert(self, errors, image_path=None):
        """Enviar alerta por Telegram"""
        if not self.bot or not TELEGRAM_CHAT_ID:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"🚨 ALERTA - Error en Impresión 3D\n"
            message += f"Fecha: {timestamp}\n"
            message += f"Errores detectados: {len(errors)}\n\n"
            
            for i, error in enumerate(errors, 1):
                message += f"{i}. {error['class']} - Confianza: {error['confidence']:.2f}\n"
            
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            
            # Enviar imagen si está disponible
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as photo:
                    await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            logger.info("Alerta enviada por Telegram")
        except Exception as e:
            logger.error(f"Error enviando alerta: {e}")

# Instancia global del detector
detector = ErrorDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/detect', methods=['POST'])
def detect_errors():
    """Endpoint para detección de errores"""
    try:
        # Verificar que se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se envió imagen"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Archivo vacío"}), 400
        
        # Procesar imagen
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a array numpy para OpenCV
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detectar errores
        errors = detector.detect_errors(image_np)
        
        # Si hay errores, enviar alerta
        if errors:
            # Guardar imagen temporalmente
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"uploads/error_{timestamp}.jpg"
            cv2.imwrite(image_path, image_np)
            
            # Enviar alerta en hilo separado
            def send_alert():
                asyncio.run(detector.send_telegram_alert(errors, image_path))
            
            thread = threading.Thread(target=send_alert)
            thread.start()
        
        return jsonify({
            "errors_detected": len(errors),
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error en endpoint detect: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Página de inicio"""
    return """
    <html>
    <head><title>Monitor de Impresión 3D</title></head>
    <body>
        <h1>Sistema de Detección de Errores en Impresión 3D</h1>
        <p>Servidor funcionando correctamente</p>
        <p>Endpoints disponibles:</p>
        <ul>
            <li>GET /health - Estado del servidor</li>
            <li>POST /detect - Detección de errores</li>
        </ul>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))