from abc import ABC, abstractmethod
import os
import json
from typing import Optional, Dict, Any

class LLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        pass



class GeminiLLMService(LLMService):
    def __init__(self, api_key: str = None, model: str = "gemini-3-flash-preview"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Enable convert_system_message_to_human since Gemini doesn't support SystemMessage
            self.client = ChatGoogleGenerativeAI(
                google_api_key=self.api_key, 
                model=self.model,
                convert_system_message_to_human=True
            )
        except ImportError:
            print("Warning: langchain-google-genai not installed.")
            self.client = None

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        if not self.client:
             return "Error: No Gemini Client available. Please install langchain-google-genai."
        
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = []
        if system_prompt:
            # SystemMessage will be auto-converted to HumanMessage by the client
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.client.invoke(messages)
            content = response.content
            if isinstance(content, list):
                # Handle multimodal content (list of dicts or strings)
                final_text = ""
                for part in content:
                    if isinstance(part, str):
                        final_text += part
                    elif isinstance(part, dict) and "text" in part:
                        final_text += part["text"]
                    else:
                        final_text += str(part)
                return final_text
            return str(content)
        except Exception as e:
            return f"Error calling Gemini: {e}"


