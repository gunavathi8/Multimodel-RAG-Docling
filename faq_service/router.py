from fastapi import APIRouter, HTTPException

from faq_service.generator import FaqGenerationService
from faq_service.schemas import FaqGenerationRequest
from src.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/faq", tags=["faq-generation"])


@router.post("/generate-document")
def generate_faq_document(req: FaqGenerationRequest) -> dict:
    logger.info("FAQ endpoint called | provider=%s", req.llm_provider)
    try:
        service = FaqGenerationService()
        return service.generate(req)
    except Exception as e:
        logger.exception("FAQ generation failed")
        raise HTTPException(status_code=500, detail=f"FAQ generation failed: {e}") from e

