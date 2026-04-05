"""
Vercel Serverless Function - Health Check
Simple health check endpoint
"""
import time

def handler(request, response):
    """Vercel Python function handler for health check"""
    
    return response.json({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'Humanoid Robotics AI Textbook API'
    })
