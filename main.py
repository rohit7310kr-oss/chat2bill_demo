import os
import json
import base64
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from rapidfuzz import process, fuzz
import jellyfish

import db_manager as db

app = FastAPI(title="Voice & Vision AI Billing System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is missing.")

client = Groq(api_key=GROQ_API_KEY)

class GenericBillPayload(BaseModel):
    items: List[dict] = []
    grand_total: float = 0.0

class TextBillingPayload(BaseModel):
    text: str

SYSTEM_PROMPT = """You are a POS billing assistant. Convert spoken or transcribed billing commands (English, Hindi, or Hinglish) into structured JSON.

ACTION RULES:
- "add": Triggered by words like "add", "put", "daal do", "plus", or default when no action verb is specified.
- "remove": Triggered by words like "remove", "hata do", "delete", "minus".
  * If quantity is specified (e.g., "remove 3 kg potato"): action: "remove", quantity: 3.0, quantity_specified: true
  * If quantity is omitted (e.g., "remove potato"): action: "remove", quantity: 1.0, quantity_specified: false
- "set": Triggered by words like "make it 5", "set to 3", "5 kar do": action: "set", quantity: 5.0, quantity_specified: true
- "remove_all": Triggered by words like "clear cart", "sab hata do", "remove everything": action: "remove_all", query_name: "all"

QUANTITY RULES:
- Convert spoken numbers and units to floats (e.g., "ek"=1.0, "do"=2.0, "aadha"=0.5, "dhai"=2.5, "3kg"=3.0, "500g"=0.5, "100ml"=0.1).
- Default quantity is 1.0 if not explicitly mentioned.

STRICT JSON OUTPUT FORMAT:
{
  "items": [
    {
      "query_name": "item name",
      "quantity": 1.0,
      "quantity_specified": true,
      "action": "add"
    }
  ]
}"""

VISION_PROMPT = """Analyze the provided image (receipt, shelf, or products). Extract all visible items and their quantities for POS billing.

Respond STRICTLY with JSON adhering to this schema:
{
  "items": [
    {
      "query_name": "item name detected in image",
      "quantity": 1.0,
      "action": "add"
    }
  ]
}"""

def match_inventory(extracted_items: List[dict]):
    inventory = db.load_inventory()
    inv_names = [item["name"] for item in inventory]
    confirmed, unmatched = [], []

    for entry in extracted_items:
        query = str(entry.get("query_name", "")).strip().lower()
        qty = float(entry.get("quantity", 1.0))
        action = entry.get("action", "add")
        qty_specified = bool(entry.get("quantity_specified", True))

        # Handle explicit clear cart command
        if action == "remove_all":
            confirmed.append({
                "item_id": "ALL",
                "inventory_name": "All Cart Items",
                "quantity": 0,
                "unit_price": 0,
                "total_price": 0,
                "action": "remove_all",
                "quantity_specified": True,
                "confidence_score": 100.0
            })
            continue

        if not query or not inv_names:
            unmatched.append({"spoken_name": query, "quantity": qty, "action": action})
            continue

        # Stage 1: Weighted Ratio Match
        match, score, index = process.extractOne(query, inv_names, scorer=fuzz.WRatio)

        # Stage 2: Token Set Ratio Fallback
        if score < 60.0:
            match_ts, score_ts, index_ts = process.extractOne(query, inv_names, scorer=fuzz.token_set_ratio)
            if score_ts > score:
                score, index = score_ts, index_ts

        # Stage 3: Phonetic Metaphone Matching
        if score < 60.0:
            query_phonetic = jellyfish.metaphone(query)
            for i, inv_item in enumerate(inv_names):
                inv_phonetic = jellyfish.metaphone(inv_item)
                if query_phonetic and inv_phonetic and query_phonetic == inv_phonetic:
                    score = 75.0
                    index = i
                    break

        if score >= 55.0:
            db_item = inventory[index]
            confirmed.append({
                "item_id": db_item["id"],
                "inventory_name": db_item["name"],
                "quantity": qty,
                "unit_price": db_item["price"],
                "total_price": round(db_item["price"] * qty, 2),
                "action": action,
                "quantity_specified": qty_specified,
                "confidence_score": round(score, 2)
            })
        else:
            unmatched.append({"spoken_name": query, "quantity": qty, "action": action})

    return confirmed, unmatched

def run_llm_pipeline(transcript_text: str):
    if not transcript_text.strip():
        return {
            "status": "success",
            "debug": {"qwen_output": None},
            "transcript": "",
            "confirmed_items": [],
            "unmatched_items": []
        }

    qwen_completion = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input Command: {transcript_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    extracted_json = json.loads(qwen_completion.choices[0].message.content)
    confirmed, unmatched = match_inventory(extracted_json.get("items", []))

    return {
        "status": "success",
        "debug": {"qwen_output": extracted_json},
        "transcript": transcript_text,
        "confirmed_items": confirmed,
        "unmatched_items": unmatched
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/v1/voice-billing-audio")
async def process_audio_billing(file: UploadFile = File(...)):
    """Primary high-accuracy audio transcription endpoint using Whisper Large v3 Turbo"""
    try:
        audio_bytes = await file.read()
        suffix = os.path.splitext(file.filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Transcribe with Groq Whisper Large v3 Turbo using inventory vocabulary hints
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                prompt="Billing voice commands in English, Hindi, and Hinglish. Common items: tamatar, pyaaz, aalu, doodh, atta, chawal, sugar, oil, vitamin c, wireless mouse, charger, 1kg, 2kg, 500g, 0.5kg, aadha kilo, dhai kilo.",
                response_format="text",
                temperature=0.0
            )
        os.remove(tmp_path)

        transcript_text = str(transcription).strip()
        return run_llm_pipeline(transcript_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice-billing-text")
async def process_text_billing(payload: TextBillingPayload):
    try:
        return run_llm_pipeline(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/image-billing")
async def process_image_billing(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = file.content_type or "image/jpeg"

        vision_completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        extracted_json = json.loads(vision_completion.choices[0].message.content)
        confirmed, unmatched = match_inventory(extracted_json.get("items", []))

        return {
            "status": "success",
            "transcript": "📷 Image Item Scan",
            "confirmed_items": confirmed,
            "unmatched_items": unmatched
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/bills/checkout")
async def checkout_bill(payload: GenericBillPayload):
    return {"status": "success", "bill": db.save_bill(items=payload.items, grand_total=payload.grand_total)}

@app.post("/api/v1/bills/clear-and-save")
async def clear_and_save_bill(payload: GenericBillPayload):
    if not payload.items:
        return {"status": "skipped", "message": "Cart was already empty."}
    return {"status": "success", "bill": db.save_bill(items=payload.items, grand_total=payload.grand_total, status="cleared")}