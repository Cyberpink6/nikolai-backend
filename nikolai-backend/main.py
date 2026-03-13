"""
NIKOLAI — Backend API v2.0
===========================
FastAPI + clasificador TF-IDF/SGD
Endpoints: /classify, /respond, /alarm, /reminder, /tasks, /tts, /health
Personalidad: casual y directo (como un amigo)
Deploy: Render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pickle, json, os, io, random, re
from datetime import datetime
from typing import Optional

# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Nikolai API",
    description="Backend del asistente virtual Nikolai",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# CARGAR MODELO
# ══════════════════════════════════════════════════════════════

MODEL_PATH     = "nikolai_classifier.pkl"
LABEL_MAP_PATH = "nikolai_label_map.json"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    print("✅ Modelo y etiquetas cargados.")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    model, label_map = None, None

# ══════════════════════════════════════════════════════════════
# ALMACENAMIENTO EN MEMORIA (reemplazar por DB en producción)
# ══════════════════════════════════════════════════════════════

alarms_db    = []   # lista de dicts {id, time, label, active}
reminders_db = []   # lista de dicts {id, text, time, active}
tasks_db     = []   # lista de dicts {id, text, done, created_at}
_id_counter  = {"alarm": 0, "reminder": 0, "task": 0}

def new_id(entity: str) -> int:
    _id_counter[entity] += 1
    return _id_counter[entity]

# ══════════════════════════════════════════════════════════════
# PLANTILLAS DE RESPUESTA — personalidad casual y directa
# ══════════════════════════════════════════════════════════════

TEMPLATES = {
    "advice": [
        "Mi consejo: empieza por lo más pequeño. El movimiento crea motivación, no al revés.",
        "Honestamente, lo que funciona es dividir el problema. ¿Qué es lo primero que puedes hacer ahora mismo?",
        "A veces la mejor respuesta es descansar. No todo se resuelve empujando más fuerte.",
        "Prueba esto: escribe el problema en papel. Ver las cosas fuera de tu cabeza ayuda un montón.",
    ],
    "opinion": [
        "Mira, es un tema donde hay de todo. Yo diría que depende mucho del contexto.",
        "Interesante pregunta. Creo que no hay una respuesta única, pero te puedo decir cómo lo veo yo.",
        "Mi opinión: hay que mirar los dos lados antes de sacar conclusiones.",
        "Sin mentirte, eso varía mucho según la situación. ¿Qué parte te interesa más?",
    ],
    "emotional": [
        "Oye, lo que sientes es válido. ¿Quieres contarme qué está pasando?",
        "Entiendo. Esos días existen y son parte del proceso. ¿Puedo ayudarte en algo concreto?",
        "A veces solo necesitas sacar lo que llevas dentro. Aquí estoy.",
        "Normal sentirse así. No tienes que resolverlo todo hoy.",
    ],
    "meta": [
        "Soy Nikolai, tu asistente. No soy humano, pero tampoco soy frío — intento ser útil de verdad.",
        "Buena pregunta. Soy un asistente entrenado para ayudarte con tareas, alarmas, recordatorios y también para escucharte.",
        "Soy Nikolai. Clasifico lo que me dices y hago lo que puedo. Sin humo, sin rollos.",
        "Me llamo Nikolai. Fui entrenado desde cero para ser directo y útil. ¿Qué necesitas?",
    ],
    "humor": [
        "¡Hey! ¿Qué tal? Por aquí listo para lo que necesites.",
        "Hola, aquí estoy. ¿En qué te puedo echar una mano?",
        "¡Buenas! Di lo que tienes, que para eso estoy.",
        "Ey, ¿todo bien? Cuéntame.",
    ],
    "knowledge": [
        "Esa es buena pregunta. Mis datos son limitados, pero lo que sé te lo digo directo.",
        "Interesante tema. Dame un poco más de contexto y te explico lo que sé.",
        "Sobre eso puedo contarte algo, aunque para profundidad te recomiendo buscar más fuentes.",
    ],
    "tech": [
        "Buena pregunta técnica. Voy al grano: ¿me das más detalles de lo que intentas hacer?",
        "En programación casi siempre hay varias formas. Dime el contexto y te digo la más directa.",
        "Eso tiene solución. Cuéntame el stack que usas y te oriento.",
    ],
    "system": [
        "Esa acción la tiene que ejecutar tu dispositivo directamente. Mándame el comando desde la app.",
        "Para ajustes del sistema necesito que la app lo gestione en tu teléfono.",
    ],
    "unknown": [
        "No entendí bien. ¿Me lo puedes decir de otra forma?",
        "Mmm, no capté. Intenta de nuevo.",
        "Eso no me quedó claro. ¿Puedes ser más específico?",
    ],
}

def get_response(intent: str, confidence: float) -> str:
    if confidence < 0.35:
        return random.choice(TEMPLATES["unknown"])
    templates = TEMPLATES.get(intent, TEMPLATES["unknown"])
    return random.choice(templates)

# ══════════════════════════════════════════════════════════════
# MODELOS DE DATOS (Pydantic)
# ══════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    text: str

class AlarmRequest(BaseModel):
    action: str            # "set" | "cancel" | "list"
    time: Optional[str] = None   # "HH:MM" o "07:30"
    label: Optional[str] = "Alarma"
    alarm_id: Optional[int] = None

class ReminderRequest(BaseModel):
    action: str            # "set" | "cancel" | "list"
    text: Optional[str] = None
    time: Optional[str] = None
    reminder_id: Optional[int] = None

class TaskRequest(BaseModel):
    action: str            # "add" | "done" | "list" | "delete"
    text: Optional[str] = None
    task_id: Optional[int] = None

class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "es"

# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

# ── /health ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0"
    }

@app.get("/")
async def root():
    return {"mensaje": "Nikolai está activo. 🟢"}

# ── /classify ──────────────────────────────────────────────────
@app.post("/classify")
async def classify_intent(request: QueryRequest):
    if not model or not label_map:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

    probs     = model.predict_proba([text])[0]
    label_idx = probs.argmax()
    label     = model.classes_[label_idx]
    confidence = round(float(probs[label_idx]), 3)

    top3 = [
        {"intent": model.classes_[i], "prob": round(float(probs[i]), 3)}
        for i in probs.argsort()[::-1][:3]
    ]

    return {
        "input": text,
        "intent": label,
        "confidence": confidence,
        "is_functional": label_map.get(label, {}).get("functional", False),
        "description": label_map.get(label, {}).get("description", "Sin descripción"),
        "top3": top3,
    }

# ── /respond ───────────────────────────────────────────────────
@app.post("/respond")
async def respond(request: QueryRequest):
    """
    Endpoint principal de la app:
    Clasifica el texto y devuelve la respuesta de Nikolai lista para mostrar.
    Para intenciones funcionales, retorna instrucción para que la app actúe.
    Para intenciones conversacionales, retorna texto de respuesta.
    """
    if not model or not label_map:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto vacío.")

    probs      = model.predict_proba([text])[0]
    label      = model.classes_[probs.argmax()]
    confidence = round(float(probs[probs.argmax()]), 3)
    is_func    = label_map.get(label, {}).get("functional", False)

    # Respuesta conversacional
    reply = get_response(label, confidence) if not is_func else None

    # Para intenciones funcionales, la app debe llamar al endpoint específico
    action_required = None
    if is_func:
        action_map = {
            "alarm_set":       {"action": "alarm_set",       "hint": "Llama a POST /alarm con action=set"},
            "alarm_cancel":    {"action": "alarm_cancel",    "hint": "Llama a POST /alarm con action=cancel"},
            "alarm_query":     {"action": "alarm_query",     "hint": "Llama a POST /alarm con action=list"},
            "reminder_set":    {"action": "reminder_set",    "hint": "Llama a POST /reminder con action=set"},
            "reminder_cancel": {"action": "reminder_cancel", "hint": "Llama a POST /reminder con action=cancel"},
            "task_add":        {"action": "task_add",        "hint": "Llama a POST /tasks con action=add"},
            "task_query":      {"action": "task_query",      "hint": "Llama a POST /tasks con action=list"},
            "task_done":       {"action": "task_done",       "hint": "Llama a POST /tasks con action=done"},
            "system":          {"action": "system",          "hint": "Ejecutar acción nativa en el dispositivo"},
        }
        action_required = action_map.get(label)

    return {
        "input":           text,
        "intent":          label,
        "confidence":      confidence,
        "is_functional":   is_func,
        "reply":           reply,
        "action_required": action_required,
    }

# ── /alarm ─────────────────────────────────────────────────────
@app.post("/alarm")
async def manage_alarm(req: AlarmRequest):
    if req.action == "set":
        if not req.time:
            raise HTTPException(status_code=400, detail="Se requiere el campo 'time' (HH:MM).")
        # Validar formato básico
        if not re.match(r"^\d{1,2}:\d{2}$", req.time):
            raise HTTPException(status_code=400, detail="Formato de hora inválido. Usa HH:MM.")
        alarm = {
            "id":     new_id("alarm"),
            "time":   req.time,
            "label":  req.label or "Alarma",
            "active": True,
            "created_at": datetime.utcnow().isoformat(),
        }
        alarms_db.append(alarm)
        return {
            "ok": True,
            "message": f"Alarma programada para las {req.time}. ✅",
            "alarm": alarm,
        }

    elif req.action == "cancel":
        if req.alarm_id:
            before = len(alarms_db)
            alarms_db[:] = [a for a in alarms_db if a["id"] != req.alarm_id]
            removed = before - len(alarms_db)
        else:
            # Cancelar todas
            removed = len(alarms_db)
            alarms_db.clear()
        return {
            "ok": True,
            "message": f"{removed} alarma(s) cancelada(s).",
            "removed": removed,
        }

    elif req.action == "list":
        active = [a for a in alarms_db if a["active"]]
        return {
            "ok": True,
            "count": len(active),
            "alarms": active,
        }

    raise HTTPException(status_code=400, detail="Acción no válida. Usa: set | cancel | list")

# ── /reminder ──────────────────────────────────────────────────
@app.post("/reminder")
async def manage_reminder(req: ReminderRequest):
    if req.action == "set":
        if not req.text:
            raise HTTPException(status_code=400, detail="Se requiere 'text' para el recordatorio.")
        reminder = {
            "id":         new_id("reminder"),
            "text":       req.text,
            "time":       req.time or None,
            "active":     True,
            "created_at": datetime.utcnow().isoformat(),
        }
        reminders_db.append(reminder)
        time_str = f" a las {req.time}" if req.time else ""
        return {
            "ok": True,
            "message": f"Recordatorio guardado{time_str}: '{req.text}' ✅",
            "reminder": reminder,
        }

    elif req.action == "cancel":
        if req.reminder_id:
            before = len(reminders_db)
            reminders_db[:] = [r for r in reminders_db if r["id"] != req.reminder_id]
            removed = before - len(reminders_db)
        else:
            removed = len(reminders_db)
            reminders_db.clear()
        return {
            "ok": True,
            "message": f"{removed} recordatorio(s) eliminado(s).",
            "removed": removed,
        }

    elif req.action == "list":
        active = [r for r in reminders_db if r["active"]]
        return {
            "ok": True,
            "count": len(active),
            "reminders": active,
        }

    raise HTTPException(status_code=400, detail="Acción no válida. Usa: set | cancel | list")

# ── /tasks ─────────────────────────────────────────────────────
@app.post("/tasks")
async def manage_tasks(req: TaskRequest):
    if req.action == "add":
        if not req.text:
            raise HTTPException(status_code=400, detail="Se requiere 'text' para la tarea.")
        task = {
            "id":         new_id("task"),
            "text":       req.text,
            "done":       False,
            "created_at": datetime.utcnow().isoformat(),
        }
        tasks_db.append(task)
        return {
            "ok": True,
            "message": f"Tarea agregada: '{req.text}' ✅",
            "task": task,
        }

    elif req.action == "done":
        if not req.task_id:
            raise HTTPException(status_code=400, detail="Se requiere 'task_id' para marcar como hecha.")
        task = next((t for t in tasks_db if t["id"] == req.task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail="Tarea no encontrada.")
        task["done"] = True
        return {
            "ok": True,
            "message": f"Tarea '{task['text']}' marcada como hecha. ✅",
            "task": task,
        }

    elif req.action == "list":
        pending  = [t for t in tasks_db if not t["done"]]
        return {
            "ok": True,
            "pending_count": len(pending),
            "tasks": tasks_db,
        }

    elif req.action == "delete":
        if not req.task_id:
            raise HTTPException(status_code=400, detail="Se requiere 'task_id'.")
        before = len(tasks_db)
        tasks_db[:] = [t for t in tasks_db if t["id"] != req.task_id]
        removed = before - len(tasks_db)
        return {
            "ok": True,
            "message": f"{removed} tarea(s) eliminada(s).",
        }

    raise HTTPException(status_code=400, detail="Acción no válida. Usa: add | done | list | delete")

# ── /tts ───────────────────────────────────────────────────────
@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Síntesis de voz server-side usando gTTS.
    Devuelve audio MP3 como stream.
    Requiere: pip install gtts
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="gTTS no instalado. Ejecuta: pip install gtts"
        )

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto vacío.")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Texto demasiado largo (máx 500 caracteres).")

    try:
        tts = gTTS(text=text, lang=req.lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=nikolai_tts.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en TTS: {str(e)}")
