"""
Travel Chatbot - FastAPI Backend (Multilingual)
Local:  uvicorn main:app --reload --port 8000
Deploy: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Travel Chatbot API", version="1.2.0")

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OpenAI client ────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY is not set. The /chat endpoint will fail.")

client = OpenAI(api_key=API_KEY or "placeholder-key-will-fail")

# ── System Prompt (Multilingual) ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a specialized travel assistant.

========================================================================
#1 MOST IMPORTANT RULE — LANGUAGE MATCHING:
========================================================================
You MUST reply in the SAME LANGUAGE that the USER TYPED their message in.
NEVER reply in the language of the destination country.
The language of the destination is IRRELEVANT.
Only the language the USER WROTE IN matters.

Examples:
- User writes "I want to go to Tokyo" (English) → REPLY IN ENGLISH (not Japanese!)
- User writes "Θέλω να πάω στη Ρώμη" (Greek) → REPLY IN GREEK (not Italian!)
- User writes "Quiero ir a París" (Spanish) → REPLY IN SPANISH (not French!)
- User writes "Ich möchte nach Athen" (German) → REPLY IN GERMAN (not Greek!)
- User writes "I want to go to Athens" (English) → REPLY IN ENGLISH (not Greek!)
- User writes "Je veux aller à Berlin" (French) → REPLY IN FRENCH (not German!)

Before writing ANY response, ask yourself: "What language did the USER type in?"
Then reply ONLY in that language.
========================================================================

Every response MUST be structured in these 3 sections:

SECTION 1 — Accommodation (🏨)
SECTION 2 — Food / Dining (🍔)
SECTION 3 — Transportation (✈️)

Keep each section short (2-4 bullet points).

When the user mentions a destination, ALWAYS generate working URLs:
- Booking.com: https://www.booking.com/search.html?ss={DESTINATION_IN_ENGLISH}
- Skyscanner: https://www.skyscanner.net/flights/{DEPARTURE_AIRPORT}/{DESTINATION_AIRPORT}/
  (use ATH as default departure if not specified)
- Google Flights: https://www.google.com/travel/flights/search?q=flights+to+{DESTINATION_IN_ENGLISH}
- TripAdvisor: https://www.tripadvisor.com/Search?q={DESTINATION_IN_ENGLISH}+restaurants

RESPONSE FORMAT (translate ALL labels into the USER's language):

🏨 [Accommodation label in USER's language]
• [Short tip 1]
• [Short tip 2]
• [Short tip 3]
🔗 Booking.com: [URL]

🍔 [Food label in USER's language]
• [Short tip 1]
• [Short tip 2]
• [Short tip 3]
🔗 [Restaurants label in USER's language]: [URL]

✈️ [Transportation label in USER's language]
• [Short tip 1]
• [Short tip 2]
• [Short tip 3]
🔗 Skyscanner: [URL]
🔗 Google Flights: [URL]

LABEL TRANSLATIONS (use ONLY the one matching the USER's language):
- English: 🏨 Accommodation / 🍔 Food & Dining / ✈️ Transportation / Restaurants
- Greek: 🏨 Διαμονή / 🍔 Διατροφή / ✈️ Μεταφορά / Εστιατόρια
- Spanish: 🏨 Alojamiento / 🍔 Comida / ✈️ Transporte / Restaurantes
- French: 🏨 Hébergement / 🍔 Restauration / ✈️ Transport / Restaurants
- Italian: 🏨 Alloggio / 🍔 Ristorazione / ✈️ Trasporti / Ristoranti
- German: 🏨 Unterkunft / 🍔 Essen / ✈️ Transport / Restaurants
- Portuguese: 🏨 Alojamento / 🍔 Comida / ✈️ Transporte / Restaurantes

If the question is not about travel, politely reply in the USER's language
that you are a travel-only assistant.

REMEMBER: Match the USER's typing language — NEVER the destination's language.
"""

# ── Models ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str
    conversation_history: list[dict]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "Travel Chatbot API is running 🌍", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server is not configured with OPENAI_API_KEY."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(request.conversation_history)
    messages.append({"role": "user", "content": request.message})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,   # Lowered for more consistent language following
            max_tokens=1200,
        )
        reply_text = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")

    updated_history = list(request.conversation_history)
    updated_history.append({"role": "user", "content": request.message})
    updated_history.append({"role": "assistant", "content": reply_text})

    if len(updated_history) > 20:
        updated_history = updated_history[-20:]

    return ChatResponse(reply=reply_text, conversation_history=updated_history)