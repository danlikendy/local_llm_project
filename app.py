#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import io
import json
import sqlite3
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

# Fix console encoding for Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
CORS(app)

# Конфигурация
DATABASE_PATH = "chat_history.db"

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Получение истории разговора
def get_conversation_history(session_id, limit=10):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_message, ai_response, timestamp 
        FROM conversations 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (session_id, limit))
    history = cursor.fetchall()
    conn.close()
    return history

# Сохранение сообщения в базу данных
def save_message(session_id, user_message, ai_response):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (session_id, user_message, ai_response)
        VALUES (?, ?, ?)
    ''', (session_id, user_message, ai_response))
    conn.commit()
    conn.close()

# Voicaj LLM теперь обрабатывает все самостоятельно

# Импорт нашей гибридной LLM с обучением
from hybrid_voicaj_llm import HybridVoicajLLM

# Создаем экземпляр гибридной Voicaj LLM
voicaj_llm = HybridVoicajLLM()

# Обработка сообщений
def process_message(message, history=None, json_mode=False):
    try:
        # Выбираем режим обработки
        if json_mode:
            # Используем нашу Voicaj LLM для JSON режима
            print(f"DEBUG: Processing message: {message[:50]}...")
            
            # Сначала проверим, какие типы обнаружены
            detected_types = voicaj_llm._detect_types(message.lower())
            print(f"DEBUG: Detected types: {detected_types}")
            
            result = voicaj_llm.analyze_text(message)
            print(f"DEBUG: Voicaj LLM returned {len(result)} objects")
            print(f"DEBUG: Object types: {[obj['type'] for obj in result]}")
            return result
        else:
            # Обычный режим - простой ответ
            return f"Получено сообщение: {message}"
            
    except Exception as e:
        print(f"DEBUG: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Ensure proper encoding for Russian text
        request.charset = 'utf-8'
        data = request.get_json(force=True)
        message = data.get('message', '').strip()
        json_mode = data.get('json_mode', False)  # Получаем режим из запроса
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get conversation history (increased limit for more context)
        session_id = session.get('session_id', str(uuid.uuid4()))
        history = get_conversation_history(session_id, limit=20)
        
        # Send request to Voicaj LLM
        response = process_message(message, history, json_mode)
        
        print(f"DEBUG: process_message returned type: {type(response)}")
        print(f"DEBUG: process_message returned {len(response) if isinstance(response, list) else 1} objects")
        if isinstance(response, list):
            print(f"DEBUG: Response is a list with {len(response)} items")
            for i, item in enumerate(response):
                print(f"DEBUG: Item {i}: type={item.get('type', 'unknown')}")
        
        # Save to database (convert response to string for storage)
        response_str = json.dumps(response) if isinstance(response, (dict, list)) else str(response)
        save_message(session_id, message, response_str)
        
        # Определяем тип ответа
        response_type = 'structured_json' if json_mode else 'text'
        
        return jsonify({
            'response': response,
            'session_id': session_id,
            'type': response_type,
            'json_mode': json_mode
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    session_id = session.get('session_id', str(uuid.uuid4()))
    history = get_conversation_history(session_id, limit=50)  # Increased limit
    
    formatted_history = []
    for user_msg, ai_msg, timestamp in reversed(history):
        # Try to parse AI response as JSON for better display
        try:
            ai_parsed = json.loads(ai_msg) if isinstance(ai_msg, str) else ai_msg
        except:
            ai_parsed = ai_msg
            
        formatted_history.append({
            'user': user_msg,
            'ai': ai_parsed,
            'timestamp': timestamp
        })
    
    return jsonify({'history': formatted_history})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    session_id = session.get('session_id', str(uuid.uuid4()))
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'История очищена'})

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Принимает обратную связь для обучения модели"""
    try:
        data = request.get_json(force=True)
        user_input = data.get('user_input', '')
        model_output = data.get('model_output', [])
        feedback = data.get('feedback', '')
        
        if not user_input or not feedback:
            return jsonify({'error': 'Необходимы user_input и feedback'}), 400
        
        # Используем новую систему улучшения
        improved_output = voicaj_llm.improve_from_feedback(user_input, model_output, feedback)
        
        return jsonify({
            'message': 'Обратная связь принята и модель улучшена',
            'improved_output': improved_output
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({'models': models})
        else:
            return jsonify({'error': 'Не удалось получить список моделей'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    print("Starting local AI assistant...")
    print(f"Web interface will be available at: http://localhost:5000")
    print("Using Hybrid Voicaj LLM model (Rule-based + Neural Network)")
    print(f"Database: {DATABASE_PATH}")
    print("=" * 50)
    
    # Запуск в режиме доступности из сети
    app.run(host='0.0.0.0', port=5000, debug=True)
