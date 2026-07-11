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

    # BULLETPROOF PROMPT: Defines exact internal schemas for arrays and dicts to prevent ANY key mismatches
    prompt = """
    You are an expert multilingual data extraction assistant. Listen to the provided audio file, which describes the statistical profile and metadata of a dataset.

    CRITICAL AUTOGRADER RULES:
    1. VALIDITY: Output ONLY valid JSON. No conversational text.
    2. MULTILINGUAL: If a column name is spoken in a non-English language (e.g., Korean "점수", Japanese), output the EXACT native script. DO NOT translate.
    3. COLUMNS ARRAY: Store exact column names as strings in the "columns" list.
    4. DICTIONARY FIELDS (mean, std, variance, min, max, median, mode, range, allowed_values, value_range): The keys inside these dictionaries MUST be the exact column names. The values are the extracted data.
    5. CORRELATION ARRAY (CRITICAL): Every object inside the "correlation" array MUST contain EXACTLY three keys: "x", "y", and "type".
       - "x": The exact name of the first column.
       - "y": The exact name of the second column.
       - "type": The correlation type (e.g., "positive", "negative", "pearson").
       - Example: [{"x": "age", "y": "점수", "type": "positive"}]
       - NEVER use keys like "column1" or "column2".

    If a metric is not mentioned, leave the container empty (0, [], or {}).
    
    TEMPLATE:
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
