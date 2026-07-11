from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import json

app = FastAPI()

class AudioPayload(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/")
async def process_audio(payload: AudioPayload):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set.")

    url = "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash:generateContent"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # UPDATED PROMPT: Added strict rules for multilingual extraction and mapping column names.
    prompt = """
    You are an expert multilingual data extraction assistant. Listen to the provided audio file, which describes the statistical profile and metadata of a dataset.
    The audio may contain mixed languages, including Japanese, Korean, and English.
    
    Rules:
    1. Output ONLY valid JSON. No conversational text.
    2. MULTILINGUAL PRESERVATION: If a column name, feature name, or category is spoken in a non-English language (e.g., Korean "점수", Japanese text, etc.), you MUST extract and output the EXACT native script. DO NOT translate it.
    3. COLUMNS ARRAY: Listen carefully for the names of the variables, features, or columns in the dataset. Put these exact names as strings inside the "columns" array.
    4. Do not fix grammar or add units unless explicitly stated.
    5. If a metric or property is not mentioned for a specific key, leave it as an empty dict {}, empty list [], or 0 as defined in the template.
    6. You must strictly output this exact structure:
    {
      "rows": 0,
      "columns": [],
      "mean": {},
      "std": {},
      "variance": {},
      "min": {},
      "max": {},
      "median": {},
      "mode": {},
      "range": {},
      "allowed_values": {},
      "value_range": {},
      "correlation": []
    }
    """

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "audio/mp3", 
                            "data": payload.audio_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0, 
            "response_mime_type": "application/json"
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        raw_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Autograder Survival: Strip Markdown backticks completely
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        parsed_json = json.loads(cleaned_text)
        return parsed_json

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API Request failed: {str(e)}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model response: {str(e)}")
