#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Fix console encoding for Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class VoicajTrainer:
    """Система обучения Voicaj LLM на примерах"""
    
    def __init__(self):
        self.training_data = []
        self.current_date = datetime.now()
        
    def add_training_example(self, user_input: str, expected_output: List[Dict[str, Any]], 
                           feedback: str = ""):
        """Добавляет пример для обучения"""
        example = {
            "input": user_input,
            "expected": expected_output,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        self.training_data.append(example)
        print(f"✅ Добавлен пример обучения: {user_input[:50]}...")
        
    def analyze_feedback(self, user_input: str, model_output: List[Dict[str, Any]], 
                        user_feedback: str) -> Dict[str, Any]:
        """Анализирует обратную связь и предлагает улучшения"""
        improvements = {
            "title_improvements": [],
            "description_improvements": [],
            "date_improvements": [],
            "tag_improvements": [],
            "priority_improvements": []
        }
        
        # Анализ обратной связи
        feedback_lower = user_feedback.lower()
        
        if "описание" in feedback_lower or "мало информации" in feedback_lower:
            improvements["description_improvements"].append("Делать описания более конкретными и информативными")
            
        if "дата" in feedback_lower or "завтра" in feedback_lower:
            improvements["date_improvements"].append("Правильно вычислять даты относительно текущего времени")
            
        if "заголовок" in feedback_lower or "название" in feedback_lower:
            improvements["title_improvements"].append("Создавать более точные и описательные заголовки")
            
        return improvements
        
    def generate_improved_output(self, user_input: str, original_output: List[Dict[str, Any]], 
                                improvements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерирует улучшенный вывод на основе обратной связи"""
        improved_output = []
        
        for obj in original_output:
            improved_obj = obj.copy()
            
            # Улучшение заголовков
            if improvements["title_improvements"]:
                improved_obj["title"] = self._improve_title(user_input, obj["type"])
                
            # Улучшение описаний
            if improvements["description_improvements"]:
                improved_obj["description"] = self._improve_description(user_input, obj["type"])
                
            # Улучшение дат
            if improvements["date_improvements"]:
                improved_obj["dueDate"] = self._improve_date(user_input)
                
            improved_output.append(improved_obj)
            
        return improved_output
        
    def _improve_title(self, user_input: str, task_type: str) -> str:
        """Улучшает заголовок на основе контекста"""
        input_lower = user_input.lower()
        
        if task_type == "task":
            if "встретиться с родителями" in input_lower:
                return "Встреча с родителями"
            elif "врач" in input_lower:
                return "Визит к врачу"
            elif "купить" in input_lower:
                return "Покупки"
            elif "программировать" in input_lower:
                return "Изучение программирования"
            else:
                return "Задача"
                
        elif task_type == "mood_entry":
            if "отлично" in input_lower:
                return "Отличное настроение"
            elif "устал" in input_lower:
                return "Усталость"
            elif "подавлен" in input_lower:
                return "Подавленность"
            else:
                return "Запись настроения"
                
        elif task_type == "habit":
            if "программировать" in input_lower:
                return "Изучение программирования"
            elif "бегать" in input_lower:
                return "Утренний бег"
            elif "читать" in input_lower:
                return "Ежедневное чтение"
            else:
                return "Новая привычка"
                
        return obj.get("title", "Запись")
        
    def _improve_description(self, user_input: str, task_type: str) -> str:
        """Улучшает описание на основе контекста"""
        input_lower = user_input.lower()
        
        if task_type == "task":
            if "встретиться с родителями" in input_lower:
                return "Встретиться с родителями для общения и поддержания семейных связей"
            elif "врач" in input_lower:
                return "Записаться на прием к врачу для консультации и обследования"
            elif "программировать" in input_lower:
                return "Начать изучение программирования для развития профессиональных навыков"
            else:
                return "Выполнить поставленную задачу"
                
        elif task_type == "mood_entry":
            if "отлично" in input_lower:
                return "Чувствую себя отлично, полон энергии и позитива"
            elif "устал" in input_lower:
                return "Испытываю усталость и нуждаюсь в отдыхе"
            else:
                return "Запись о текущем эмоциональном состоянии"
                
        elif task_type == "habit":
            if "программировать" in input_lower:
                return "Регулярно изучать программирование для профессионального развития"
            elif "бегать" in input_lower:
                return "Ежедневно бегать по утрам для поддержания физической формы"
            else:
                return "Развить новую полезную привычку"
                
        return "Описание задачи"
        
    def _improve_date(self, user_input: str) -> str:
        """Улучшает дату на основе контекста"""
        input_lower = user_input.lower()
        
        if "завтра" in input_lower:
            tomorrow = self.current_date + timedelta(days=1)
            return tomorrow.strftime("%Y-%m-%d %H:%M")
        elif "послезавтра" in input_lower:
            day_after = self.current_date + timedelta(days=2)
            return day_after.strftime("%Y-%m-%d %H:%M")
        elif "сегодня" in input_lower:
            return self.current_date.strftime("%Y-%m-%d %H:%M")
        else:
            return self.current_date.strftime("%Y-%m-%d %H:%M")
            
    def save_training_data(self, filename: str = "voicaj_training_data.json"):
        """Сохраняет данные обучения в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Данные обучения сохранены в {filename}")
        
    def load_training_data(self, filename: str = "voicaj_training_data.json"):
        """Загружает данные обучения из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            print(f"📂 Загружено {len(self.training_data)} примеров обучения")
        except FileNotFoundError:
            print(f"⚠️ Файл {filename} не найден, начинаем с пустого набора")
            
    def print_training_summary(self):
        """Выводит сводку по обучению"""
        print(f"\n📊 СВОДКА ПО ОБУЧЕНИЮ:")
        print(f"Всего примеров: {len(self.training_data)}")
        
        if self.training_data:
            print(f"Последний пример: {self.training_data[-1]['input'][:50]}...")
            print(f"Время последнего обновления: {self.training_data[-1]['timestamp']}")

# Пример использования
if __name__ == "__main__":
    trainer = VoicajTrainer()
    
    # Загружаем существующие данные
    trainer.load_training_data()
    
    # Пример с обратной связью
    user_input = "сегодня я чувствую себя отлично, поставь задачу на завтра встретиться с родителями, а также хочу начать программировать"
    
    # Оригинальный вывод модели (плохой)
    original_output = [
        {
            "description": "Выполнить поставленную задачу",
            "dueDate": "2025-10-08 23:08",
            "priority": "high",
            "tags": ["работа", "семья", "настроение"],
            "title": "Встреча",
            "type": "task"
        },
        {
            "description": "Запись о текущем настроении",
            "priority": "high",
            "tags": ["работа", "семья", "настроение"],
            "title": "Запись настроения",
            "type": "mood_entry"
        }
    ]
    
    # Обратная связь пользователя
    feedback = "очень плохое описание задачи, мало информации, почему в описании ты не написал что встреча и что именно с родителями? также почему ты поставил 8 октября, если завтра 7"
    
    # Анализируем обратную связь
    improvements = trainer.analyze_feedback(user_input, original_output, feedback)
    print(f"\n🔍 АНАЛИЗ ОБРАТНОЙ СВЯЗИ:")
    for category, items in improvements.items():
        if items:
            print(f"{category}: {items}")
    
    # Генерируем улучшенный вывод
    improved_output = trainer.generate_improved_output(user_input, original_output, improvements)
    print(f"\n✨ УЛУЧШЕННЫЙ ВЫВОД:")
    for i, obj in enumerate(improved_output):
        print(f"{i+1}. {obj['title']} - {obj['description']}")
        print(f"   Дата: {obj.get('dueDate', 'N/A')}")
    
    # Сохраняем пример обучения
    trainer.add_training_example(user_input, improved_output, feedback)
    trainer.save_training_data()
    trainer.print_training_summary()
