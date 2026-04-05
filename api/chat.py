"""
Vercel Serverless Function - Chat Endpoint
Wraps the existing RAG agent for Vercel deployment
"""
import sys
import os
import json
import time

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

def handler(request, response):
    """Vercel Python function handler for chat endpoint"""
    
    # Only allow POST requests
    if request.method != 'POST':
        return response.json({
            'status': 'error',
            'error': 'Method not allowed'
        }, status_code=405)
    
    try:
        # Parse request body
        body = request.json()
        question = body.get('question', '')
        selected_text = body.get('selected_text')
        
        if not question:
            return response.json({
                'status': 'error',
                'error': 'Question is required'
            }, status_code=400)
        
        # Import and run the RAG agent
        from agent.rag_agent import run_agent
        
        start_time = time.time()
        result = run_agent(
            question=question,
            selected_text=selected_text,
            use_cache=True
        )
        
        elapsed = time.time() - start_time
        
        return response.json({
            'status': 'ok',
            'data': {
                'answer': result['answer'],
                'sources': result.get('sources', [])
            },
            'metadata': {
                'response_time': elapsed
            }
        })
        
    except Exception as e:
        return response.json({
            'status': 'error',
            'error': str(e)
        }, status_code=500)
