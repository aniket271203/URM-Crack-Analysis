#!/usr/bin/env python3
"""
Test script to verify Gemini AI setup for crack analysis.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_gemini_setup():
    """Test if Gemini AI is properly configured."""
    print("Testing Gemini AI setup...")
    
    # Test 1: Check if the package is installed
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package is installed")
    except ImportError:
        print("❌ google-generativeai package is not installed")
        print("   Install with: pip install google-generativeai")
        return False
    
    # Test 2: Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Set with: export GEMINI_API_KEY='your_api_key_here'")
        return False
    else:
        print("✅ GEMINI_API_KEY environment variable is set")
    
    # Test 3: Try to configure Gemini
    try:
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured successfully")
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {str(e)}")
        return False
    
    # Test 4: Try to create a model
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini model created successfully")
    except Exception as e:
        print(f"❌ Failed to create Gemini model: {str(e)}")
        return False
    
    # Test 5: Try a simple text generation
    try:
        response = model.generate_content("Hello, this is a test.")
        if response and hasattr(response, 'text'):
            print("✅ Gemini text generation working")
            print(f"   Response: {response.text[:50]}...")
        else:
            print("❌ Gemini response is empty or invalid")
            return False
    except Exception as e:
        print(f"❌ Failed to generate content: {str(e)}")
        return False
    
    print("\n🎉 All tests passed! Gemini AI is ready for crack analysis.")
    return True

if __name__ == "__main__":
    success = test_gemini_setup()
    sys.exit(0 if success else 1)
