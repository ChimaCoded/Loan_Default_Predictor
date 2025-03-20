import uvicorn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("WELCOME TO PESWA LOAN DEFAULT PREDICTOR!")
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    from app.api import app
    uvicorn.run(app, host="0.0.0.0", port=8000) 