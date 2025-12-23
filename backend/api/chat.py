from agents.rag_agent import RAGAgent
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

from .models import ChatRequest, ChatResponse, HealthResponse

from database.postgres_client import PostgresService
from utils.logging import setup_logging

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, summary="Chat with the RAG agent")
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    Main chat endpoint that processes user queries using RAG.

    This endpoint:
    1. Takes a user query and optional parameters
    2. Uses the RAG agent to retrieve relevant context
    3. Generates a response using OpenAI's API
    4. Logs the interaction to Neon Postgres
    5. Returns the response with sources and metadata
    """
    start_time = time.time()

    try:
        logger.info(f"Processing chat request for user: {request.user_id}")

        # Get services from app state
        rag_agent = req.app.state.rag_agent
        postgres_service = req.app.state.postgres_service

        # Process the query using the RAG agent
        response_data = await rag_agent.process_query(
            query=request.query,
            user_selected_text=request.user_selected_text,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        response_time = time.time() - start_time

        # Prepare response
        response = ChatResponse(
            response=response_data["response"],
            sources=response_data["sources"],
            response_time=response_time,
            query=request.query
        )

        # Log to Neon Postgres
        await postgres_service.log_chat_interaction(
            query=request.query,
            response=response_data["response"],
            context=str(response_data.get("context", "")),
            sources=response_data["sources"],
            response_time_ms=response_time * 1000,
            user_id=request.user_id
        )

        logger.info(f"Chat request completed successfully. Response time: {response_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        response_time = time.time() - start_time

        # Get services from app state for error logging
        postgres_service = req.app.state.postgres_service

        # Log error to Neon Postgres
        await postgres_service.log_error(
            query=request.query,
            error_message=str(e),
            response_time_ms=response_time * 1000,
            user_id=request.user_id
        )

        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/query", response_model=ChatResponse, summary="Alternative query endpoint with more options")
async def query_endpoint(request: ChatRequest, req: Request):
    """
    Alternative query endpoint with additional options for advanced use cases.
    Similar to /chat but may include additional parameters for fine-tuning.
    """
    return await chat_endpoint(request, req)

@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check(req: Request):
    """
    Health check endpoint to verify the service is running and all dependencies are accessible.

    Returns the status of the service and the timestamp of the check.
    """
    try:
        # Get services from app state
        rag_agent = req.app.state.rag_agent
        postgres_service = req.app.state.postgres_service

        # Check if all services are available
        services_status = []

        # Check RAG agent
        if rag_agent.is_healthy():
            services_status.append("RAG agent: OK")
        else:
            services_status.append("RAG agent: UNHEALTHY")

        # Check database connection
        if await postgres_service.is_healthy():
            services_status.append("Database: OK")
        else:
            services_status.append("Database: UNHEALTHY")

        logger.info(f"Health check completed: {', '.join(services_status)}")

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Additional utility endpoints
@router.get("/stats", summary="Get usage statistics")
async def get_stats(req: Request):
    """
    Get usage statistics from the database.
    """
    try:
        # Get postgres service from app state
        postgres_service = req.app.state.postgres_service

        stats = await postgres_service.get_usage_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Note: Middleware should be added at the app level, not router level
# This middleware functionality is handled in the main server.py file