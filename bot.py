import logging
import os
import base64
import json
import io
import asyncio
import concurrent.futures
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jinja2 import Template
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ConversationHandler, filters, ContextTypes
from telegram.constants import ChatType
import google.generativeai as genai
import requests

# ========== НАСТРОЙКИ ==========
BOT_TOKEN = "8946461977:AAE9qPY3zt611wpuzrE3X4KgFGi3M7zOmYY"
GEMINI_API_KEY = "AIzaSyB_VYvENU02jpHVSk3v3BBAf0eFVoiOOTY"

genai.configure(api_key=GEMINI_API_KEY)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== КАСКАД ==========
cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
if not os.path.exists(cascade_path):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(cascade_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        logger.warning(f"Каскад: {e}")
face_cascade = cv2.CascadeClassifier(cascade_path)

# ========== ДАННЫЕ ==========
DATA_FILE = "user_data.json"
GROUP_DATA_FILE = "group_data.json"
battle_sessions = {}
group_battle_sessions = {}

# Состояния для ConversationHandler
LANG, GENDER, AGE = range(3)

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_group_data():
    if os.path.exists(GROUP_DATA_FILE):
        with open(GROUP_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_group_data(data):
    with open(GROUP_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ========== ЛОКАЛИЗАЦИЯ ==========
LANG_DICT = {
    "ru": {
        "start_private": "🌟 <b>PSL Face Analyzer</b>\n\n🔍 Оценка по шкале PSL 1–8\n🌟 Сравнение со знаменитостями\n📈 График прогресса\n⚔️ Баттл лиц\n\nВ группах: .rate .battle .help .leaderboard",
        "start_group": "🤖 Бот активен! Используй .help для списка команд.",
        "check": "📸 Отправь фото для анализа.",
        "profile": "📊 <b>Профиль:</b>\nПроверок: {checks}\nЛучший PSL: {best}\nКатегория: {last_cat}",
        "profile_empty": "Профиль пуст. Отправь фото!",
        "battle_prompt": "⚔️ Отправь <b>первое</b> фото.",
        "battle_first_ok": "✅ Первое лицо: PSL {psl}\nОтправь второе.",
        "battle_winner": "🏆 Первое лицо!" if "{winner}" == "1" else ("🏆 Второе лицо!" if "{winner}" == "2" else "🤝 Ничья!"),
        "leaderboard_title": "🏆 ТОП-10 PSL",
        "history_need_more": "📈 Нужно минимум 2 оценки.",
        "advice_prompt": "📸 Отправь фото для советов!",
        "processing": "⏳ Обработка фото 20%",
        "evaluating": "🔍 Оценка лица 40%",
        "aspects": "📊 Оценка по аспектам 60%",
        "preparing": "📝 Подготовка ответа 80%",
        "done": "✅ Ответ готов! 100%",
        "error": "❌ Ошибка анализа. Попробуй другое фото.",
        "face_not_found": "❌ Лицо не найдено.",
        "group_rate_reply": "📸 Ответь командой <b>.rate</b> на сообщение с фото!",
        "group_battle_reply": "⚔️ Ответь <b>.battle</b> на фото соперника и отправь своё!",
        "group_battle_send": "⚔️ Теперь отправь <b>своё фото</b> для баттла!",
        "group_leaderboard_empty": "🏆 В этой группе ещё нет оценок.",
        "group_no_psl": "У тебя ещё нет оценок в этой группе.",
        "language_select": "Выберите язык / Choose language:",
        "gender_select": "Выберите пол:",
        "age_prompt": "Введите возраст (5–100):",
        "profile_updated": "Профиль сохранён! Добро пожаловать.",
        "help_text": "<b>🤖 PSL Bot — команды группы:</b>\n\n<b>.rate</b> — ответь на сообщение с фото, чтобы оценить\n<b>.battle</b> — ответь на фото для баттла, затем отправь своё\n<b>.leaderboard</b> — топ-10 участников группы\n<b>.mypsl</b> — твой последний PSL\n<b>.help</b> — это сообщение"
    },
    "en": {
        "start_private": "🌟 <b>PSL Face Analyzer</b>\n\n🔍 PSL 1–8 rating\n🌟 Celebrity look‑alike\n📈 Progress graph\n⚔️ Face battle\n\nIn groups: .rate .battle .help .leaderboard",
        "start_group": "🤖 Bot active! Use .help for commands.",
        "check": "📸 Send a photo for analysis.",
        "profile": "📊 <b>Profile:</b>\nChecks: {checks}\nBest PSL: {best}\nCategory: {last_cat}",
        "profile_empty": "Profile empty. Send a photo!",
        "battle_prompt": "⚔️ Send the <b>first</b> photo.",
        "battle_first_ok": "✅ First face: PSL {psl}\nSend the second.",
        "battle_winner": "🏆 First face wins!" if "{winner}" == "1" else ("🏆 Second face wins!" if "{winner}" == "2" else "🤝 Draw!"),
        "leaderboard_title": "🏆 TOP 10 PSL",
        "history_need_more": "📈 Need at least 2 ratings.",
        "advice_prompt": "📸 Send a photo for advice!",
        "processing": "⏳ Processing photo 20%",
        "evaluating": "🔍 Evaluating face 40%",
        "aspects": "📊 Aspect analysis 60%",
        "preparing": "📝 Preparing response 80%",
        "done": "✅ Response ready! 100%",
        "error": "❌ Analysis error. Try another photo.",
        "face_not_found": "❌ Face not found.",
        "group_rate_reply": "📸 Reply with <b>.rate</b> to a photo message!",
        "group_battle_reply": "⚔️ Reply with <b>.battle</b> to an opponent's photo and send yours!",
        "group_battle_send": "⚔️ Now send <b>your photo</b> for the battle!",
        "group_leaderboard_empty": "🏆 No ratings in this group yet.",
        "group_no_psl": "You have no ratings in this group yet.",
        "language_select": "Выберите язык / Choose language:",
        "gender_select": "Choose gender:",
        "age_prompt": "Enter age (5–100):",
        "profile_updated": "Profile saved! Welcome.",
        "help_text": "<b>🤖 PSL Bot — group commands:</b>\n\n<b>.rate</b> — reply to a photo to rate\n<b>.battle</b> — reply to a photo for battle, then send yours\n<b>.leaderboard</b> — top‑10 group members\n<b>.mypsl</b> — your last PSL\n<b>.help</b> — this message"
    }
}

def get_text(user_id, key, **kwargs):
    data = load_data()
    uid = str(user_id)
    lang = data.get(uid, {}).get('language', 'ru')
    text = LANG_DICT.get(lang, LANG_DICT['ru']).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    return text

# ========== HTML-ШАБЛОН (мультиязычный) ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PSL Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: #f5f6fa;
            color: #1a1a2e;
            line-height: 1.6;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 720px;
            background: #ffffff;
            border-radius: 32px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.06);
            overflow: hidden;
        }
        .header {
            padding: 40px 30px 20px;
            text-align: center;
            border-bottom: 1px solid #f0f0f0;
        }
        .header h1 { font-weight: 600; font-size: 2rem; color: #111; }
        .header .subtitle { color: #666; font-size: 0.95rem; }
        .score-panel {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            padding: 30px 20px;
            background: #fafbfc;
            border-bottom: 1px solid #f0f0f0;
        }
        .score-circle {
            width: 120px; height: 120px;
            border-radius: 50%;
            background: #fff;
            border: 3px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .score-value { font-size: 3.5rem; font-weight: 700; color: #1e293b; }
        .score-meta { text-align: left; }
        .category { font-size: 1.4rem; font-weight: 600; color: #1e293b; }
        .psl-label { font-size: 0.9rem; color: #64748b; text-transform: uppercase; }
        .photo-section { text-align: center; padding: 20px; }
        .photo-section img { max-width: 240px; border-radius: 20px; }
        .section { padding: 24px 30px; border-bottom: 1px solid #f0f0f0; }
        .section h2 { font-weight: 600; font-size: 1.2rem; color: #1e293b; margin-bottom: 16px; }
        .aspect-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        .aspect-item { background: #f8fafc; padding: 14px 16px; border-radius: 14px; }
        .aspect-name { font-size: 0.9rem; color: #475569; margin-bottom: 6px; }
        .bar-track { height: 8px; background: #e2e8f0; border-radius: 10px; }
        .bar-fill { height: 100%; background: #3b82f6; border-radius: 10px; }
        .aspect-value { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
        .advantages-list { list-style: none; }
        .advantages-list li { padding: 8px 0; font-size: 0.95rem; color: #334155; }
        .advantages-list li::before { content: "•"; color: #3b82f6; font-weight: bold; margin-right: 8px; }
        .lookalike-row { display: flex; gap: 16px; flex-wrap: wrap; }
        .lookalike-card { background: #f8fafc; border-radius: 14px; padding: 12px 18px; }
        .lookalike-card .name { font-weight: 600; color: #1e293b; }
        .lookalike-card .percent { font-size: 0.9rem; color: #3b82f6; }
        .emotion { font-size: 1.1rem; color: #334155; }
        .comparison { display: flex; align-items: center; justify-content: center; gap: 20px; flex-wrap: wrap; }
        .comparison img { max-width: 220px; border-radius: 16px; }
        .footer { text-align: center; padding: 20px; color: #94a3b8; font-size: 0.8rem; }
        .god-badge {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: #1e293b;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            display: inline-block;
            margin-top: 8px;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid #f0f0f0; }
        th { font-weight: 600; color: #64748b; font-size: 0.85rem; }
    </style>
</head>
<body><div class="container">{{ content }}</div></body>
</html>
"""

# ========== ФУНКЦИИ ==========
def image_to_base64(image_bgr):
    try:
        success, buffer = cv2.imencode('.jpg', image_bgr)
        return base64.b64encode(buffer).decode('utf-8') if success else ""
    except:
        return ""

def analyze_face_full(image_bgr, user_id=None):
    """Анализ лица через Gemini с учётом пола, возраста и эмоций"""
    max_retries = 3
    # Загружаем профиль
    data = load_data()
    uid = str(user_id) if user_id else None
    profile = data.get(uid, {}) if uid else {}
    gender = profile.get('gender', 'male')
    age = profile.get('age', 25)
    lang = profile.get('language', 'ru')
    lang_prompt = "Russian" if lang == 'ru' else "English"

    for attempt in range(max_retries):
        try:
            h, w = image_bgr.shape[:2]
            max_size = 512
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image_bgr = cv2.resize(image_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            success, encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success: return None
            image_bytes = encoded.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            model = genai.GenerativeModel('gemini-2.5-flash')
            # Промпт зависит от пола
            if gender == 'female':
                category_names = {
                    "sub": "Subhuman",
                    "low": "Low-Tier Becky",
                    "normie": "Becky",
                    "high": "High-Tier Becky",
                    "chadlite": "Stacylite",
                    "chad": "Stacy",
                    "true": "True Stacy",
                    "god": "PSL GODDESS"
                }
            else:
                category_names = {
                    "sub": "Subhuman",
                    "low": "Low-Tier Normie",
                    "normie": "Normie",
                    "high": "High-Tier Normie",
                    "chadlite": "Chadlite",
                    "chad": "Chad",
                    "true": "True Adam",
                    "god": "PSL GOD"
                }

            prompt = f"""
Ты эксперт по оценке внешности согласно строгой шкале PSL (от 1 до 8).
Оцени лицо на фото. Известно: пол – {gender}, возраст – {age} лет.
Параметры (0–100):
- Симметрия
- Челюсть
- Глаза
- Кожа
- Гармония
- Диморфизм

Также оцени видимую эмоцию на лице (например: уверенность, радость, грусть, нейтрально, удивление). Дай одно слово.

Определи итоговый PSL (число от 1 до 8, одно значение после запятой).
Категории (для пола {gender}):
1.0–2.4: {category_names['sub']}
2.5–3.4: {category_names['low']}
3.5–4.9: {category_names['normie']}
5.0–6.4: {category_names['high']}
6.5–7.4: {category_names['chadlite']}
7.5–7.9: {category_names['chad']}
8.0: {category_names['true']}
>8.0: {category_names['god']}

Сравни с 1–2 известными людьми, выдели 3 преимущества.
Верни СТРОГО ТОЛЬКО JSON (ключи на английском):
{{
  "psl": 7.8,
  "category": "Chad",
  "symmetry": 88,
  "jaw": 91,
  "eyes": 78,
  "skin": 80,
  "harmony": 85,
  "dimorphism": 90,
  "advantages": ["Adv1", "Adv2", "Adv3"],
  "lookalike": [{{"name": "Name", "percent": 89}}],
  "emotion": "confident"
}}
Ответь на {lang_prompt}.
"""
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                request_options={"timeout": 90}
            )
            text = response.text.strip()
            if text.startswith('```json'): text = text[7:-3].strip()
            elif text.startswith('```'): text = text[3:-3].strip()
            data = json.loads(text)

            return {
                "psl_score": data['psl'],
                "category": data['category'],
                "aspects": [
                    {"name": "🎭 Симметрия", "value": data['symmetry']},
                    {"name": "💎 Челюсть", "value": data['jaw']},
                    {"name": "👁️ Глаза", "value": data['eyes']},
                    {"name": "🌟 Кожа", "value": data['skin']},
                    {"name": "🎨 Гармония", "value": data['harmony']},
                    {"name": "👤 Диморфизм", "value": data['dimorphism']},
                ],
                "advantages": [{"emoji": "•", "text": adv} for adv in data.get('advantages', [])],
                "lookalike": data.get('lookalike', []),
                "emotion": data.get('emotion', 'neutral'),
                "photo_base64": image_base64,
                "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "lang": lang
            }
        except Exception as e:
            logger.warning(f"Попытка {attempt+1}: {e}")
            if attempt < max_retries - 1: time.sleep(2)
            else: return None

def generate_single_report(report_data, user_id):
    data = load_data()
    uid = str(user_id)
    lang = data.get(uid, {}).get('language', 'ru')
    psl = report_data['psl_score']
    cat = report_data['category']
    god = '<span class="god-badge">PSL GOD</span>' if psl > 8.0 else ('<span class="god-badge">True Adam/Stacy</span>' if psl == 8.0 else '')
    emotion = report_data.get('emotion', 'neutral')

    # Локализованные заголовки
    t = lambda key: LANG_DICT[lang].get(key, key)
    content = f"""
    <div class="header"><h1>PSL Face Report</h1><div class="subtitle">Персональная оценка</div></div>
    <div class="photo-section"><img src="data:image/jpeg;base64,{report_data['photo_base64']}" alt="Фото"></div>
    <div class="score-panel">
        <div class="score-circle"><span class="score-value">{psl}</span></div>
        <div class="score-meta"><div class="category">{cat}</div><div class="psl-label">шкала PSL 1–8</div>{god}</div>
    </div>
    <div class="section"><h2>😐 Эмоция</h2><p class="emotion">{emotion}</p></div>
    """
    if report_data.get('lookalike'):
        content += '<div class="section"><h2>🌟 Похожие знаменитости</h2><div class="lookalike-row">'
        for lk in report_data['lookalike']:
            content += f'<div class="lookalike-card"><span class="name">{lk["name"]}</span> <span class="percent">{lk["percent"]}%</span></div>'
        content += '</div></div>'

    content += f"""
    <div class="section"><h2>📊 Детализация</h2><div class="aspect-grid">
        {''.join(f'<div class="aspect-item"><div class="aspect-name">{a["name"]}</div><div class="bar-track"><div class="bar-fill" style="width:{a["value"]}%"></div></div><div class="aspect-value">{a["value"]}%</div></div>' for a in report_data['aspects'])}
    </div></div>
    <div class="section"><h2>✅ Сильные стороны</h2><ul class="advantages-list">
        {''.join(f'<li>{adv["text"]}</li>' for adv in report_data['advantages'])}
    </ul></div>
    <div class="footer">PSL Analyzer · {report_data["date"]}</div>
    """
    tmpl = Template(HTML_TEMPLATE)
    return tmpl.render(content=content, lang=lang)

# ========== ПРОФИЛЬ (ConversationHandler) ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    uid = str(user_id)
    data = load_data()
    profile = data.get(uid, {})

    if update.message.chat.type != ChatType.PRIVATE:
        await update.message.reply_text("🤖 Бот активен! Используй .help для списка команд.")
        return ConversationHandler.END

    if not profile or 'language' not in profile:
        keyboard = [
            [InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
             InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")]
        ]
        await update.message.reply_text(
            "Выберите язык / Choose language:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return LANG
    else:
        await show_main_menu(update, context, user_id)
        return ConversationHandler.END

async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang = query.data.split("_")[1]
    context.user_data['lang'] = lang
    # Сохраняем язык сразу
    user_id = query.from_user.id
    data = load_data()
    uid = str(user_id)
    if uid not in data:
        data[uid] = {}
    data[uid]['language'] = lang
    save_data(data)

    if lang == "ru":
        keyboard = [
            [InlineKeyboardButton("Мужской", callback_data="gender_male"),
             InlineKeyboardButton("Женский", callback_data="gender_female")]
        ]
        await query.edit_message_text("Выберите пол:", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        keyboard = [
            [InlineKeyboardButton("Male", callback_data="gender_male"),
             InlineKeyboardButton("Female", callback_data="gender_female")]
        ]
        await query.edit_message_text("Choose gender:", reply_markup=InlineKeyboardMarkup(keyboard))
    return GENDER

async def gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    gender = query.data.split("_")[1]
    context.user_data['gender'] = gender
    user_id = query.from_user.id
    data = load_data()
    uid = str(user_id)
    data[uid]['gender'] = gender
    save_data(data)

    lang = data[uid].get('language', 'ru')
    if lang == "ru":
        await query.edit_message_text("Введите возраст (5–100):")
    else:
        await query.edit_message_text("Enter age (5–100):")
    return AGE

async def age_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    uid = str(user_id)
    text = update.message.text.strip()
    if not text.isdigit() or not (5 <= int(text) <= 100):
        lang = context.user_data.get('lang', 'ru')
        if lang == "ru":
            await update.message.reply_text("Пожалуйста, введите число от 5 до 100.")
        else:
            await update.message.reply_text("Please enter a number between 5 and 100.")
        return AGE
    age = int(text)
    data = load_data()
    data[uid]['age'] = age
    save_data(data)

    lang = data[uid].get('language', 'ru')
    if lang == "ru":
        await update.message.reply_text("Профиль сохранён! Добро пожаловать.")
    else:
        await update.message.reply_text("Profile saved! Welcome.")
    await show_main_menu(update, context, user_id)
    return ConversationHandler.END

async def show_main_menu(update, context, user_id):
    keyboard = [
        [InlineKeyboardButton("👤 Профиль", callback_data="profile")],
        [InlineKeyboardButton("🔍 Проверить внешность", callback_data="check")],
        [InlineKeyboardButton("⚔️ Батл внешности", callback_data="battle")],
        [InlineKeyboardButton("🏆 Топ игроков", callback_data="leaderboard")],
        [InlineKeyboardButton("📈 История оценок", callback_data="history")],
        [InlineKeyboardButton("💡 Советы по улучшению", callback_data="advice")],
    ]
    data = load_data()
    uid = str(user_id)
    lang = data.get(uid, {}).get('language', 'ru')
    text = LANG_DICT[lang]["start_private"]
    if update.callback_query:
        await update.callback_query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    uid = str(user_id)
    data = load_data()
    lang = data.get(uid, {}).get('language', 'ru')

    if query.data == "check":
        await query.message.reply_text(LANG_DICT[lang]["check"])
    elif query.data == "profile":
        user = data.get(uid, {})
        if user:
            await query.message.reply_text(
                LANG_DICT[lang]["profile"].format(
                    checks=user.get('checks',0),
                    best=user.get('best_psl','—'),
                    last_cat=user.get('last_category','—')
                ),
                parse_mode="HTML"
            )
        else:
            await query.message.reply_text(LANG_DICT[lang]["profile_empty"])
    elif query.data == "battle":
        battle_sessions[user_id] = {"photos": [], "stage": 1}
        await query.message.reply_text(LANG_DICT[lang]["battle_prompt"], parse_mode="HTML")
    elif query.data == "leaderboard":
        html = generate_leaderboard_html(data, lang)
        filename = f"lb_{user_id}.html"
        with open(filename, "w", encoding="utf-8") as f: f.write(html)
        with open(filename, "rb") as f:
            await query.message.reply_document(document=f, filename="PSL_Top10.html")
        os.remove(filename)
    elif query.data == "history":
        user = data.get(uid, {})
        buf = generate_history_graph(user, lang)
        if buf:
            await query.message.reply_photo(photo=buf, caption="📈 Прогресс PSL")
        else:
            await query.message.reply_text(LANG_DICT[lang]["history_need_more"])
    elif query.data == "advice":
        context.user_data['waiting_for_advice'] = True
        await query.message.reply_text(LANG_DICT[lang]["advice_prompt"])

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_type = update.message.chat.type

    if context.user_data.get('waiting_for_advice'):
        context.user_data['waiting_for_advice'] = False
        await handle_advice(update, context)
        return

    if chat_type == ChatType.PRIVATE and user_id in battle_sessions and battle_sessions[user_id]["stage"] in [1,2]:
        await handle_battle_photo(update, context)
        return

    if chat_type in [ChatType.GROUP, ChatType.SUPERGROUP] and user_id in group_battle_sessions:
        await handle_group_battle_photo(update, context)
        return

    await handle_single_photo(update, context)

async def handle_single_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    uid = str(user_id)
    chat_id = str(update.message.chat.id)
    chat_type = update.message.chat.type
    is_group = chat_type in [ChatType.GROUP, ChatType.SUPERGROUP]
    data = load_data()
    lang = data.get(uid, {}).get('language', 'ru')

    msg = await update.message.reply_text(LANG_DICT[lang]["processing"])

    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        await asyncio.sleep(0.5)
        await msg.edit_text(LANG_DICT[lang]["evaluating"])

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            report = await loop.run_in_executor(pool, analyze_face_full, img, user_id)

        if report is None:
            await msg.edit_text(LANG_DICT[lang]["error"])
            return

        await msg.edit_text(LANG_DICT[lang]["aspects"])
        await asyncio.sleep(0.3)
        await msg.edit_text(LANG_DICT[lang]["preparing"])

        html = generate_single_report(report, user_id)
        filename = f"report_{user_id}.html"
        with open(filename, "w", encoding="utf-8") as f: f.write(html)

        await msg.edit_text(LANG_DICT[lang]["done"])

        caption = f"📋 PSL: {report['psl_score']} — {report['category']}"
        if report.get('lookalike'):
            caption += f"\n🌟 Похож на: {report['lookalike'][0]['name']} ({report['lookalike'][0]['percent']}%)"
        if report.get('emotion'):
            caption += f"\n😐 Эмоция: {report['emotion']}"

        with open(filename, "rb") as f:
            if is_group:
                await update.message.reply_document(
                    document=f, filename="PSL_Report.html",
                    caption=caption,
                    reply_to_message_id=update.message.message_id
                )
            else:
                await update.message.reply_document(document=f, filename="PSL_Report.html", caption=caption)
        os.remove(filename)
        await msg.delete()

        # Сохраняем данные
        data = load_data()
        if uid not in data:
            data[uid] = {"checks": 0, "best_psl": 0, "history": []}
        data[uid]["checks"] += 1
        data[uid]["last_psl"] = report['psl_score']
        data[uid]["last_category"] = report['category']
        data[uid]["username"] = update.message.from_user.username or update.message.from_user.first_name or uid
        if report['psl_score'] > data[uid].get("best_psl", 0):
            data[uid]["best_psl"] = report['psl_score']
        data[uid]["history"].append({
            "date": datetime.now().isoformat(),
            "psl": report['psl_score'],
            "category": report['category']
        })
        save_data(data)

        if is_group:
            gdata = load_group_data()
            if chat_id not in gdata:
                gdata[chat_id] = {}
            gdata[chat_id][uid] = {
                "username": update.message.from_user.username or update.message.from_user.first_name or uid,
                "best_psl": max(report['psl_score'], gdata[chat_id].get(uid, {}).get('best_psl', 0)),
                "last_psl": report['psl_score']
            }
            save_group_data(gdata)

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await msg.edit_text(LANG_DICT[lang]["error"])

async def handle_advice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    data = load_data()
    lang = data.get(str(user_id), {}).get('language', 'ru')
    msg = await update.message.reply_text("💡 Готовлю советы...")
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, encoded = cv2.imencode('.jpg', rgb)

        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt_lang = "Russian" if lang == 'ru' else "English"
        response = model.generate_content([
            f"Дай 5 советов по улучшению внешности на {prompt_lang}. Каждый с эмодзи. Верни JSON: {{\"advices\": [\"💇 Совет 1\", ...]}}",
            {"mime_type": "image/jpeg", "data": encoded.tobytes()}
        ], request_options={"timeout": 30})
        text = response.text.strip()
        if text.startswith('```json'): text = text[7:-3].strip()
        advices = json.loads(text).get('advices', [])
        await msg.edit_text("💡 <b>Советы:</b>\n" + "\n".join(advices), parse_mode="HTML")
    except Exception as e:
        logger.error(f"Советы: {e}")
        await msg.edit_text("❌ Ошибка.")

async def handle_battle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    session = battle_sessions[user_id]
    data = load_data()
    lang = data.get(str(user_id), {}).get('language', 'ru')
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            report = await loop.run_in_executor(pool, analyze_face_full, img, user_id)

        if report is None:
            await update.message.reply_text(LANG_DICT[lang]["face_not_found"])
            del battle_sessions[user_id]
            return

        if session["stage"] == 1:
            session["photos"].append(report)
            session["stage"] = 2
            await update.message.reply_text(LANG_DICT[lang]["battle_first_ok"].format(psl=report['psl_score']))
        else:
            session["photos"].append(report)
            r1, r2 = session["photos"]
            psl1, psl2 = r1['psl_score'], r2['psl_score']
            winner = "1" if psl1 > psl2 else ("2" if psl2 > psl1 else "0")
            winner_text = LANG_DICT[lang]["battle_winner"].format(winner=winner)
            content = f"""
            <div class="header"><h1>⚔️ PSL БАТТЛ</h1></div>
            <div class="comparison">
                <div><img src="data:image/jpeg;base64,{r1['photo_base64']}"><p>PSL: {psl1}</p></div>
                <div style="font-size:2rem;">VS</div>
                <div><img src="data:image/jpeg;base64,{r2['photo_base64']}"><p>PSL: {psl2}</p></div>
            </div>
            <div style="text-align:center;font-size:1.5rem;margin:20px;">{winner_text}</div>
            """
            html = Template(HTML_TEMPLATE).render(content=content, lang=lang)
            filename = f"battle_{user_id}.html"
            with open(filename, "w", encoding="utf-8") as f: f.write(html)
            with open(filename, "rb") as f:
                await update.message.reply_document(document=f, filename="PSL_Battle.html",
                    caption=f"⚔️ {winner_text}\nPSL 1: {psl1} | PSL 2: {psl2}")
            os.remove(filename)
            del battle_sessions[user_id]
    except Exception as e:
        logger.error(f"Баттл: {e}")
        del battle_sessions[user_id]

# ========== ГРУППОВЫЕ ОБРАБОТЧИКИ ==========
async def group_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.message.chat.type
    if chat_type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        return

    text = update.message.text.lower().strip()
    chat_id = str(update.message.chat.id)
    user_id = update.message.from_user.id
    # Язык пользователя берём из его профиля
    data = load_data()
    lang = data.get(str(user_id), {}).get('language', 'ru')

    if text == ".help":
        await update.message.reply_text(LANG_DICT[lang]["help_text"], parse_mode="HTML")

    elif text == ".leaderboard":
        gdata = load_group_data()
        group = gdata.get(chat_id, {})
        if not group:
            await update.message.reply_text(LANG_DICT[lang]["group_leaderboard_empty"])
            return

        sorted_users = sorted(group.items(), key=lambda x: x[1].get('best_psl', 0), reverse=True)[:10]
        rows = ""
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for i, (uid, udata) in enumerate(sorted_users, 1):
            medal = medals.get(i, f"#{i}")
            name = udata.get('username', uid)
            psl = udata.get('best_psl', 0)
            rows += f"{medal} {name} — PSL {psl}\n"

        await update.message.reply_text(f"🏆 <b>Топ-10 группы:</b>\n\n{rows}", parse_mode="HTML")

    elif text == ".mypsl":
        gdata = load_group_data()
        group = gdata.get(chat_id, {})
        uid = str(user_id)
        user = group.get(uid, {})
        if user:
            await update.message.reply_text(
                f"📊 Твой PSL: {user.get('last_psl', '—')} (лучший: {user.get('best_psl', '—')})",
                reply_to_message_id=update.message.message_id
            )
        else:
            await update.message.reply_text(LANG_DICT[lang]["group_no_psl"])

    elif text == ".rate":
        if not update.message.reply_to_message or not update.message.reply_to_message.photo:
            await update.message.reply_text(LANG_DICT[lang]["group_rate_reply"], parse_mode="HTML", reply_to_message_id=update.message.message_id)
            return

        await update.message.reply_text("⏳ Оцениваю...")
        try:
            photo_file = await update.message.reply_to_message.photo[-1].get_file()
            image_bytes = await photo_file.download_as_bytearray()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            target_user_id = update.message.reply_to_message.from_user.id
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                report = await loop.run_in_executor(pool, analyze_face_full, img, target_user_id)

            if report is None:
                await update.message.reply_text("❌ Ошибка.")
                return

            caption = f"📋 PSL: {report['psl_score']} — {report['category']}"
            if report.get('lookalike'):
                caption += f"\n🌟 Похож на: {report['lookalike'][0]['name']} ({report['lookalike'][0]['percent']}%)"
            if report.get('emotion'):
                caption += f"\n😐 Эмоция: {report['emotion']}"

            await update.message.reply_text(
                caption,
                reply_to_message_id=update.message.reply_to_message.message_id
            )

            gdata = load_group_data()
            if chat_id not in gdata:
                gdata[chat_id] = {}
            target_uid = str(target_user_id)
            target_name = update.message.reply_to_message.from_user.username or update.message.reply_to_message.from_user.first_name or target_uid
            gdata[chat_id][target_uid] = {
                "username": target_name,
                "best_psl": max(report['psl_score'], gdata[chat_id].get(target_uid, {}).get('best_psl', 0)),
                "last_psl": report['psl_score']
            }
            save_group_data(gdata)

        except Exception as e:
            logger.error(f"Ошибка .rate: {e}")
            await update.message.reply_text("❌ Ошибка.")

    elif text == ".battle":
        if not update.message.reply_to_message or not update.message.reply_to_message.photo:
            await update.message.reply_text(LANG_DICT[lang]["group_battle_reply"], parse_mode="HTML", reply_to_message_id=update.message.message_id)
            return

        group_battle_sessions[user_id] = {
            "opponent_photo": update.message.reply_to_message.photo[-1],
            "chat_id": chat_id,
            "opponent_id": update.message.reply_to_message.from_user.id
        }
        await update.message.reply_text(LANG_DICT[lang]["group_battle_send"], parse_mode="HTML", reply_to_message_id=update.message.message_id)

async def handle_group_battle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id not in group_battle_sessions:
        return

    session = group_battle_sessions[user_id]
    data = load_data()
    lang = data.get(str(user_id), {}).get('language', 'ru')
    await update.message.reply_text("⏳ Оцениваю баттл...")

    try:
        opp_photo = session["opponent_photo"]
        opp_file = await opp_photo.get_file()
        opp_bytes = await opp_file.download_as_bytearray()
        opp_nparr = np.frombuffer(opp_bytes, np.uint8)
        opp_img = cv2.imdecode(opp_nparr, cv2.IMREAD_COLOR)

        my_photo = await update.message.photo[-1].get_file()
        my_bytes = await my_photo.download_as_bytearray()
        my_nparr = np.frombuffer(my_bytes, np.uint8)
        my_img = cv2.imdecode(my_nparr, cv2.IMREAD_COLOR)

        opponent_id = session["opponent_id"]
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            opp_report = await loop.run_in_executor(pool, analyze_face_full, opp_img, opponent_id)
            my_report = await loop.run_in_executor(pool, analyze_face_full, my_img, user_id)

        if not opp_report or not my_report:
            await update.message.reply_text("❌ Ошибка анализа.")
            del group_battle_sessions[user_id]
            return

        psl1, psl2 = opp_report['psl_score'], my_report['psl_score']
        winner = "1" if psl1 > psl2 else ("2" if psl2 > psl1 else "0")
        winner_text = LANG_DICT[lang]["battle_winner"].format(winner=winner)

        await update.message.reply_text(
            f"⚔️ <b>Результат баттла:</b>\n"
            f"Соперник: PSL {psl1} — {opp_report['category']}\n"
            f"Ты: PSL {psl2} — {my_report['category']}\n\n"
            f"{winner_text}",
            parse_mode="HTML",
            reply_to_message_id=update.message.message_id
        )

        del group_battle_sessions[user_id]
    except Exception as e:
        logger.error(f"Баттл в группе: {e}")
        del group_battle_sessions[user_id]

# ========== ОБЩИЕ ФУНКЦИИ ==========
def generate_leaderboard_html(data, lang):
    sorted_users = sorted(data.items(), key=lambda x: x[1].get('best_psl', 0), reverse=True)[:10]
    rows = ""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for i, (uid, udata) in enumerate(sorted_users, 1):
        medal = medals.get(i, f"#{i}")
        name = udata.get('username', uid)
        psl = udata.get('best_psl', 0)
        rows += f'<tr><td>{medal}</td><td>{name}</td><td>{psl}</td></tr>'
    title = LANG_DICT[lang]["leaderboard_title"]
    content = f"""
    <div class="header"><h1>{title}</h1></div>
    <table><tr><th>#</th><th>Игрок</th><th>PSL</th></tr>{rows}</table>
    <div class="footer">Обновлено: {datetime.now().strftime("%d.%m.%Y %H:%M")}</div>
    """
    return Template(HTML_TEMPLATE).render(content=content, lang=lang)

def generate_history_graph(user_data, lang):
    history = user_data.get('history', [])
    if len(history) < 2: return None
    dates = [h['date'][:10] for h in history]
    psls = [h['psl'] for h in history]
    plt.figure(figsize=(8, 4), facecolor='#f5f6fa')
    ax = plt.axes()
    ax.set_facecolor('#f5f6fa')
    ax.plot(dates, psls, marker='o', color='#3b82f6', linewidth=2, markersize=8)
    ax.fill_between(range(len(dates)), psls, alpha=0.3, color='#3b82f6')
    ax.set_ylim(0, 8.5)
    ax.set_ylabel('PSL', color='#1e293b')
    ax.set_xlabel('Дата' if lang == 'ru' else 'Date', color='#1e293b')
    ax.tick_params(colors='#1e293b')
    ax.grid(True, alpha=0.2)
    plt.title('📈 История PSL' if lang == 'ru' else '📈 PSL History', color='#1e293b')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#f5f6fa')
    buf.seek(0)
    plt.close()
    return buf

# ========== ЗАПУСК ==========
def main():
    application = Application.builder().token(BOT_TOKEN).build()

    # Conversation handler для настройки профиля
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            LANG: [CallbackQueryHandler(lang_callback, pattern="^lang_")],
            GENDER: [CallbackQueryHandler(gender_callback, pattern="^gender_")],
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age_input)]
        },
        fallbacks=[],
    )
    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
        group_message_handler
    ))

    logger.info("Бот запущен с профилями, эмоциями и языками")
    application.run_polling()

if __name__ == "__main__":
    main()