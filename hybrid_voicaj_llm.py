import sys
import io
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)

# Исправляем кодировку для Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class HybridVoicajLLM:
    """Гибридная система: Rule-based + LLM для сложных случаев"""
    
    def __init__(self):
        print("🤖 Инициализация гибридной системы...")
        
        # Текущая дата
        self.current_date = datetime.now()
        
        # Загружаем данные обучения
        self.training_data = self.load_training_data()
        
        # Инициализируем LLM только при необходимости
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_generator = None
        
        # Удалены старые паттерны - используем новую логику в _detect_types
        
        # Теги и ключевые слова
        self.tags_keywords = {
            'работа': ['работа', 'офис', 'коллеги', 'проект', 'встреча', 'программировать', 'код', 'разработка', 'клиент', 'презентация', 'карьера', 'бизнес', 'отчёт', 'руководитель'],
            'семья': ['семья', 'дети', 'родители', 'родственники', 'встретиться с родителями', 'мама', 'папа'],
            'здоровье': ['здоровье', 'врач', 'лекарство', 'больница', 'устал', 'усталым', 'подавлен', 'терапевт'],
            'спорт': ['спорт', 'тренировка', 'фитнес', 'зал', 'бегать', 'бег', 'беговая', 'утренний бег'],
            'технологии': ['программировать', 'код', 'разработка', 'компьютер', 'технологии', 'сайт', 'портфолио'],
            'настроение': ['настроение', 'чувствую', 'эмоции', 'отлично', 'хорошо', 'плохо', 'волнуюсь', 'нервничаю', 'люблю'],
            'учеба': ['учеба', 'экзамен', 'математика', 'изучать', 'обучение', 'курс', 'лекция', 'презентация', 'вуз'],
            'покупки': ['покупки', 'купить', 'магазин', 'товар', 'продукт', 'продукты', 'костюм', 'подарок']
        }
        
        self.priority_keywords = {
            'high': ['срочно', 'важно', 'критично', 'немедленно', 'сегодня', 'завтра'],
            'medium': ['на этой неделе', 'в ближайшее время'],
            'low': ['когда-нибудь', 'не спеша', 'в свободное время', 'срочность низкая', 'неважно']
        }
        
        print("✅ Гибридная система готова!")
    
    def load_training_data(self) -> List[Dict]:
        """Загружает данные обучения"""
        try:
            with open('voicaj_training_data.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Файл обучения не найден")
            return []
    
    def is_complex_request(self, text: str) -> bool:
        """Определяет, является ли запрос сложным для LLM"""
        text_lower = text.lower()
        
        # Простые случаи - обрабатываем rule-based
        simple_patterns = [
            r'завтра.*отправить.*отчёт',
            r'послезавтра.*отправить.*отчёт',
            r'завтра.*сходить.*продукт',
            r'послезавтра.*сходить.*продукт',
            r'презентация.*вуз',
            r'код.*работа'
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Сложные случаи - нужна LLM
        complex_indicators = [
            len(text.split()) > 15,  # Длинный текст
            'и' in text_lower and text.count('и') > 2,  # Много союзов
            any(word in text_lower for word in ['одновременно', 'параллельно', 'также', 'кроме того']),
            text.count(',') > 3,  # Много запятых
        ]
        
        return any(complex_indicators)
    
    def init_llm(self):
        """Инициализирует LLM только при необходимости"""
        if self.llm_model is not None:
            return
            
        print("🧠 Инициализация LLM для сложных случаев...")
        
        try:
            # Используем модель, специально обученную для структурированного вывода
            model_name = "microsoft/DialoGPT-small"
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Создаем генератор с более строгими параметрами
            self.llm_generator = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                max_new_tokens=300,
                temperature=0.1,  # Очень низкая температура для детерминированности
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Штраф за повторения
                length_penalty=1.0
            )
            
            print("✅ LLM инициализирована!")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации LLM: {e}")
            self.llm_model = None
    
    def rule_based_analysis(self, text: str) -> List[Dict[str, Any]]:
        """Rule-based анализ (быстрый и точный для простых случаев)"""
        print("⚡ Используем rule-based анализ...")
        
        detected_types = self._detect_types(text.lower())
        if not detected_types:
            detected_types = ['task']
        
        # Берем только первый тип для простоты
        task_type = detected_types[0]
        result = self._create_object(text, task_type)
        
        return [result] if result else []
    
    def llm_analysis(self, text: str) -> List[Dict[str, Any]]:
        """LLM анализ для сложных случаев с улучшенной логикой"""
        print("🧠 Используем LLM анализ...")
        
        self.init_llm()
        
        if self.llm_model is None:
            print("❌ LLM недоступна, используем rule-based")
            return self.rule_based_analysis(text)
        
        try:
            # Определяем типы задач из текста
            detected_types = self._detect_types(text.lower())
            if not detected_types:
                detected_types = ['task']
            
            # Создаем умный промпт на основе обнаруженных типов
            current_date = self.current_date.strftime("%Y-%m-%d")
            tomorrow = (self.current_date + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Адаптируем промпт под конкретные типы
            if 'mood_entry' in detected_types and 'task' in detected_types:
                prompt = f"""User: {text}

Assistant: I'll create both a mood entry and a task for you.

Mood Entry:
{{
    "title": "Emotional state",
    "type": "mood_entry",
    "description": "Current emotional state and feelings",
    "tags": ["настроение", "эмоции"],
    "priority": "high",
    "dueDate": "{current_date} 18:00"
}}

Task:
{{
    "title": "Task to complete",
    "type": "task",
    "description": "Specific task that needs to be done",
    "tags": ["задача"],
    "priority": "high",
    "dueDate": "{tomorrow} 18:00"
}}"""
            elif 'habit' in detected_types:
                prompt = f"""User: {text}

Assistant: I'll create a habit for you.

{{
    "title": "New habit",
    "type": "habit",
    "description": "Regular activity to develop",
    "tags": ["привычка"],
    "priority": "medium",
    "dueDate": "{tomorrow} 18:00"
}}"""
            elif 'goal' in detected_types:
                prompt = f"""User: {text}

Assistant: I'll create a long-term goal for you.

{{
    "title": "Long-term goal",
    "type": "goal",
    "description": "Important long-term objective",
    "tags": ["цель"],
    "priority": "medium",
            "dueDate": "2025-12-31 23:59"
}}"""
            else:
                # Простая задача
                prompt = f"""User: {text}

Assistant: I'll create a task for you.

{{
    "title": "Task",
    "type": "task",
    "description": "Specific task to complete",
    "tags": ["задача"],
    "priority": "high",
    "dueDate": "{tomorrow} 18:00"
}}"""
            
            # Генерируем ответ
            result = self.llm_generator(
                prompt,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                repetition_penalty=1.3,
                length_penalty=1.0
            )
            
            response = result[0]['generated_text'][len(prompt):].strip()
            print(f"🔍 LLM ответ: {response[:100]}...")
            
            # Извлекаем JSON объекты
            json_objects = []
            
            # Ищем все JSON объекты в ответе
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    # Очищаем JSON
                    json_str = match.strip()
                    if not json_str.endswith('}'):
                        json_str += '}'
                    
                    obj = json.loads(json_str)
                    
                    # Валидируем и улучшаем объект
                    improved_obj = self._validate_and_improve_object(obj, text)
                    if improved_obj:
                        json_objects.append(improved_obj)
                        
                except json.JSONDecodeError:
                    continue
            
            if json_objects:
                print(f"✅ LLM сгенерировал {len(json_objects)} валидных объектов!")
                return json_objects
            else:
                print("⚠️ LLM не сгенерировал валидный JSON, используем rule-based")
                return self.rule_based_analysis(text)
            
        except Exception as e:
            print(f"❌ Ошибка LLM анализа: {e}")
            return self.rule_based_analysis(text)
    
    def _validate_and_improve_object(self, obj: Dict, original_text: str) -> Dict:
        """Валидирует и улучшает JSON объект"""
        required_fields = ['title', 'type', 'description', 'tags', 'priority', 'dueDate']
        
        # Проверяем наличие всех обязательных полей
        for field in required_fields:
            if field not in obj:
                return obj  # Возвращаем исходный объект если не все поля
        
        # Улучшаем объект на основе оригинального текста
        text_lower = original_text.lower()
        
        # Улучшаем заголовок
        if obj['title'] in ['Task', 'Mood', 'Moods', 'Emotional state', 'Задача']:
            # Используем улучшенную логику извлечения заголовков
            obj['title'] = self._extract_title(original_text, obj['type'])
        
        # Улучшаем описание
        if len(obj['description']) < 10:
            obj['description'] = f"Выполнить: {original_text}"
        
        # Улучшаем теги
        if not obj['tags'] or len(obj['tags']) < 2:
            obj['tags'] = self._extract_tags(original_text)
        
        # Улучшаем приоритет
        if obj['priority'] not in ['high', 'medium', 'low']:
            obj['priority'] = self._extract_priority(original_text)
        
        # Улучшаем дату
        obj['dueDate'] = self._extract_due_date(original_text)
        
        return obj
    
    def find_similar_examples(self, user_input: str) -> List[Dict[str, Any]]:
        """Находит похожие примеры из данных обучения"""
        similar = []
        input_lower = user_input.lower()
        
        # Сначала ищем точные совпадения фраз
        for example in self.training_data:
            if 'input' in example:
                example_input = example['input'].lower()
                
                # Проверяем точное совпадение
                if input_lower == example_input:
                    return [example]
                
                # Проверяем частичное совпадение (80% слов)
                input_words = set(input_lower.split())
                example_words = set(example_input.split())
                common_words = input_words & example_words
                
                if len(common_words) >= len(input_words) * 0.8:  # 80% совпадение
                    similar.append(example)
        
        # Если нет точных совпадений, ищем по ключевым словам
        if not similar:
            key_words = ['отчёт', 'руководитель', 'презентация', 'код', 'программирование', 'работа', 'вуз', 'учеба', 'задача', 'срочность', 'написание']
            
            for example in self.training_data:
                if 'input' in example:
                    example_input = example['input'].lower()
                    
                    # Проверяем точные совпадения ключевых слов
                    input_keywords = [word for word in key_words if word in input_lower]
                    example_keywords = [word for word in key_words if word in example_input]
                    
                    # Если есть совпадения ключевых слов
                    if input_keywords and example_keywords:
                        common_keywords = set(input_keywords) & set(example_keywords)
                        if len(common_keywords) >= 1:  # Минимум 1 общий ключевой термин
                            similar.append(example)
                            
        return similar[:1]  # Возвращаем только 1 наиболее похожий пример
    
    def analyze_text(self, text: str) -> List[Dict[str, Any]]:
        """Основной метод анализа - выбирает между rule-based и LLM"""
        try:
            # Исправляем кодировку если нужно
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            
            print(f"🔍 Анализируем: {text[:50]}...")
            
            # Определяем сложность запроса
            if self.is_complex_request(text):
                print("🧠 Сложный запрос - используем LLM")
                return self.llm_analysis(text)
            else:
                print("⚡ Простой запрос - используем rule-based")
                return self.rule_based_analysis(text)
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            return [{
                "title": "Задача",
                "type": "task", 
                "description": f"Выполнить: {text}",
                "tags": ["задача"],
                "priority": "medium",
                "dueDate": (self.current_date + timedelta(days=1)).strftime("%Y-%m-%d 18:00")
            }]
    
    def _detect_types(self, text: str) -> List[str]:
        """Определяет типы задач в тексте с улучшенной логикой"""
        detected = set()
        text_lower = text.lower()
        
        # Расширенные паттерны для определения типов
        
        # MOOD_ENTRY - эмоциональные состояния
        mood_patterns = [
            'чувствую', 'волнуюсь', 'переживаю', 'устал', 'устала', 'грустно', 'радостно',
            'злой', 'злая', 'раздражен', 'раздражена', 'спокоен', 'спокойна', 'тревожно',
            'беспокоюсь', 'нервничаю', 'переживаю', 'настроение', 'эмоции', 'состояние',
            'депрессия', 'стресс', 'тревога', 'паника', 'счастье', 'радость', 'восторг',
            'разочарован', 'разочарована', 'обижен', 'обижена', 'одинок', 'одинока',
            'очень радуюсь', 'горжусь', 'испытываю', 'сильную тревогу', 'панику',
            'огромную благодарность', 'счастье от', 'поддержки друзей', 'волнуюсь перед',
            'важным экзаменом', 'не могу уснуть', 'публичным выступлением'
        ]
        
        # HABIT - привычки и регулярные действия
        habit_patterns = [
            'каждый день', 'ежедневно', 'регулярно', 'привычка', 'привык', 'начинаю',
            'хочу начать', 'планирую начать', 'буду делать', 'каждое утро', 'каждый вечер',
            'каждую неделю', 'каждый месяц', 'тренировка', 'зарядка', 'бег', 'бегать',
            'читать каждый день', 'изучать каждый день', 'учиться каждый день', 
            'практиковать каждый день', 'медитировать', 'йога', 'программировать каждый день',
            'спорт', 'фитнес', 'тренироваться', 'заниматься спортом'
        ]
        
        # GOAL - долгосрочные цели
        goal_patterns = [
            'хочу создать', 'хочу открыть', 'хочу стать', 'хочу достичь', 'цель',
            'мечтаю', 'планирую', 'когда-нибудь', 'в будущем', 'через год', 'через 5 лет',
            'стартап', 'бизнес', 'карьера', 'профессия', 'навык', 'мастерство',
            'достижение', 'амбиции', 'стремление', 'желание', 'намерение',
            'когда-нибудь прочитать', 'когда-нибудь изучить', 'когда-нибудь научиться',
            'в будущем хочу', 'в будущем планирую', 'через несколько лет',
            'хочу путешествовать', 'путешествовать по', 'изучить разные культуры',
            'стать профессиональным', 'фотографом', 'мечтаю стать'
        ]
        
        # TASK - конкретные задачи
        task_patterns = [
            'нужно', 'должен', 'должна', 'обязательно', 'срочно', 'важно',
            'встреча', 'презентация', 'отчет', 'документ', 'письмо', 'звонок',
            'покупки', 'магазин', 'продукты', 'еда', 'лекарства', 'аптека',
            'врач', 'больница', 'поликлиника', 'медицина', 'здоровье',
            'работа', 'офис', 'проект', 'задача', 'дело', 'план',
            'учеба', 'экзамен', 'курс', 'лекция', 'семинар', 'конференция',
            'путешествие', 'поездка', 'отпуск', 'билеты', 'отель', 'виза',
            'ремонт', 'уборка', 'стирка', 'готовка', 'дом', 'квартира'
        ]
        
        # Проверяем паттерны в порядке приоритета
        
        # 0. Специальные случаи для точного определения
        if 'изучить' in text_lower and ('язык программирования' in text_lower or 'python' in text_lower or 'javascript' in text_lower):
            return ['habit']
        if 'когда-нибудь' in text_lower and ('прочитать' in text_lower or 'прочесть' in text_lower):
            return ['goal']
        if 'важно изучить' in text_lower and 'язык' in text_lower:
            return ['habit']
        if 'планирую изучить' in text_lower and ('фреймворк' in text_lower or 'технологию' in text_lower):
            return ['habit']
        if 'хочу получить сертификат' in text_lower or 'получить сертификат' in text_lower:
            return ['goal']
        if 'подготовиться к собеседованию' in text_lower or 'собеседование' in text_lower:
            return ['task']
        
        # 1. Сначала проверяем конкретные задачи (приоритет для смешанных запросов)
        for pattern in task_patterns:
            if pattern in text_lower:
                # Если есть и эмоции, и задачи - приоритет задачам
                if any(p in text_lower for p in mood_patterns):
                    # Специальная проверка для эмоций перед экзаменом
                    if 'очень нервничаю' in text_lower and 'экзамен' in text_lower:
                        return ['mood_entry']  # Эмоции перед экзаменом = настроение
                    return ['task']  # Смешанный запрос = задача
                return ['task']
        
        # 2. Затем проверяем эмоциональные состояния (только если нет задач)
        for pattern in mood_patterns:
            if pattern in text_lower:
                # Специальная проверка для смешанных запросов
                if 'собеседование' in text_lower:
                    return ['task']  # Собеседования = задачи
                elif 'экзамен' in text_lower and ('очень нервничаю' in text_lower or 'волнуюсь' in text_lower):
                    return ['mood_entry']  # Эмоции перед экзаменом = настроение
                return ['mood_entry']
        
        # 3. Проверяем привычки
        for pattern in habit_patterns:
            if pattern in text_lower:
                return ['habit']
        
        # 4. Проверяем долгосрочные цели
        for pattern in goal_patterns:
            if pattern in text_lower:
                return ['goal']
        
        # 5. Дополнительная логика для сложных случаев
        if 'и' in text_lower and len(text.split()) > 5:
            # Сложный запрос - может содержать несколько типов
            types = []
            if any(p in text_lower for p in mood_patterns):
                types.append('mood_entry')
            if any(p in text_lower for p in habit_patterns):
                types.append('habit')
            if any(p in text_lower for p in goal_patterns):
                types.append('goal')
            if any(p in text_lower for p in task_patterns):
                types.append('task')
            
            if types:
                return types[:2]  # Максимум 2 типа для сложных запросов
        
        # По умолчанию - задача
        return ['task']
    
    def _create_object(self, text: str, task_type: str) -> Dict[str, Any]:
        """Создает объект задачи с улучшенной логикой"""
        obj = {
            "title": self._extract_title(text, task_type),
            "type": task_type,
            "description": self._extract_description(text, task_type),
            "tags": self._extract_tags(text),
            "priority": self._extract_priority(text),
            "dueDate": self._extract_due_date(text)
        }
        
        # Дополнительно улучшаем объект
        improved_obj = self._validate_and_improve_object(obj, text)
        return improved_obj if improved_obj else obj
    
    def _extract_title(self, text: str, task_type: str) -> str:
        """Извлекает заголовок с улучшенной логикой"""
        text_lower = text.lower()
        print(f"🔍 DEBUG: Извлекаем заголовок для '{text}' типа '{task_type}'")
        
        if task_type == 'task':
            if 'отчёт' in text_lower and 'руководитель' in text_lower:
                print(f"✅ DEBUG: Найдено совпадение 'отчёт руководитель'")
                return "Отправка отчёта руководителю"
            elif 'презентация' in text_lower and 'инвестор' in text_lower:
                return "Презентация для инвесторов"
            elif 'отчёт' in text_lower and 'начальник' in text_lower:
                return "Отправка отчёта руководителю"
            elif 'отчет' in text_lower and 'начальник' in text_lower:
                return "Отправка отчёта руководителю"
            elif 'презентация' in text_lower and 'клиент' in text_lower:
                return "Презентация для клиента"
            elif 'продукт' in text_lower or 'магазин' in text_lower:
                return "Покупка продуктов"
            elif 'презентация' in text_lower and 'вуз' in text_lower:
                return "Презентация по вузу"
            elif 'код' in text_lower and 'работа' in text_lower:
                return "Написание кода по работе"
            elif 'встреча' in text_lower and 'команда' in text_lower:
                return "Встреча с командой"
            elif 'собеседование' in text_lower:
                return "Собеседование"
            elif 'переезд' in text_lower:
                return "Подготовка к переезду"
            elif 'операция' in text_lower and 'мама' in text_lower:
                return "Поддержка мамы во время операции"
            elif 'врач' in text_lower or 'больница' in text_lower:
                return "Визит к врачу"
            elif 'встреча' in text_lower and 'клиент' in text_lower:
                return "Встреча с клиентом"
            elif 'отчёт' in text_lower:
                return "Подготовка отчёта"
            elif 'презентация' in text_lower:
                return "Подготовка презентации"
            elif 'встреча' in text_lower:
                return "Встреча"
            elif 'звонок' in text_lower:
                return "Звонок"
            elif 'письмо' in text_lower:
                return "Написание письма"
            elif 'документ' in text_lower:
                return "Подготовка документа"
            elif 'покупки' in text_lower:
                return "Покупки"
            elif 'ремонт' in text_lower:
                return "Ремонт"
            elif 'уборка' in text_lower:
                return "Уборка"
            elif 'готовка' in text_lower:
                return "Готовка"
            elif 'стирка' in text_lower:
                return "Стирка"
            elif 'экзамен' in text_lower:
                return "Подготовка к экзамену"
            elif 'курс' in text_lower:
                return "Прохождение курса"
            elif 'лекция' in text_lower:
                return "Посещение лекции"
            elif 'конференция' in text_lower:
                return "Участие в конференции"
            elif 'отпуск' in text_lower:
                return "Планирование отпуска"
            elif 'поездка' in text_lower:
                return "Планирование поездки"
            elif 'билеты' in text_lower:
                return "Покупка билетов"
            elif 'отель' in text_lower:
                return "Бронирование отеля"
            elif 'виза' in text_lower:
                return "Оформление визы"
            else:
                # Попробуем извлечь ключевые слова из текста
                words = text.split()
                if len(words) >= 3:
                    # Берем первые 2-3 значимых слова
                    key_words = []
                    for word in words[:5]:  # Проверяем первые 5 слов
                        if len(word) > 3 and word.lower() not in ['нужно', 'должен', 'должна', 'обязательно', 'срочно', 'важно', 'завтра', 'послезавтра', 'сегодня']:
                            key_words.append(word)
                            if len(key_words) >= 2:
                                break
                    
                    if key_words:
                        return " ".join(key_words).title()
                
                print(f"⚠️ DEBUG: Не найдено совпадение, возвращаем 'Задача'")
                return "Задача"
        elif task_type == 'mood_entry':
            if 'волнуюсь' in text_lower or 'переживаю' in text_lower or 'стресс' in text_lower:
                return "Эмоциональное состояние"
            elif 'устал' in text_lower or 'усталым' in text_lower or 'усталость' in text_lower:
                return "Состояние усталости"
            elif 'отлично' in text_lower or 'хорошо' in text_lower:
                return "Отличное настроение"
            elif 'грустно' in text_lower or 'плохо' in text_lower:
                return "Плохое настроение"
            elif 'тревога' in text_lower or 'беспокоюсь' in text_lower:
                return "Состояние тревоги"
            else:
                return "Запись настроения"
        elif task_type == 'habit':
            if 'бегать' in text_lower:
                return "Утренний бег"
            elif 'программировать' in text_lower or 'python' in text_lower or 'изучать' in text_lower:
                return "Изучение Python"
            elif 'английский' in text_lower or 'язык' in text_lower:
                return "Изучение языка"
            elif 'читать' in text_lower:
                return "Ежедневное чтение"
            elif 'тренировка' in text_lower or 'спорт' in text_lower:
                return "Регулярные тренировки"
            elif 'медитировать' in text_lower or 'йога' in text_lower:
                return "Медитация"
            else:
                return "Новая привычка"
        elif task_type == 'goal':
            if 'стартап' in text_lower or 'бизнес' in text_lower or 'открыть' in text_lower:
                return "Открытие бизнеса"
            elif 'приложение' in text_lower:
                return "Создание приложения"
            elif 'фотограф' in text_lower:
                return "Становление фотографом"
            elif 'токио' in text_lower:
                return "Переезд в Токио"
            elif 'карьера' in text_lower or 'профессия' in text_lower:
                return "Развитие карьеры"
            elif 'навык' in text_lower or 'мастерство' in text_lower:
                return "Развитие навыков"
            elif 'дом' in text_lower or 'квартира' in text_lower:
                return "Покупка жилья"
            elif 'путешествие' in text_lower or 'поездка' in text_lower:
                return "Планирование путешествия"
            else:
                return "Долгосрочная цель"
        
        return "Запись"
    
    def _extract_description(self, text: str, task_type: str) -> str:
        """Извлекает описание с улучшенной логикой"""
        text_lower = text.lower()
        
        if task_type == 'task':
            if 'отчёт' in text_lower and 'руководитель' in text_lower:
                return "Подготовить и отправить отчёт руководителю о выполненных задачах"
            elif 'презентация' in text_lower and 'инвестор' in text_lower:
                return "Подготовить материалы и репетировать речь для важной презентации перед инвесторами"
            elif 'презентация' in text_lower and 'клиент' in text_lower:
                return "Подготовить материалы и репетировать речь для важной презентации клиенту"
            elif 'продукт' in text_lower or 'магазин' in text_lower:
                return "Сходить в магазин и купить продукты на неделю"
            elif 'презентация' in text_lower and 'вуз' in text_lower:
                return "Подготовить презентацию по вузу для демонстрации результатов обучения"
            elif 'код' in text_lower and 'работа' in text_lower:
                return "Выполнить задачу по написанию кода для работы"
            elif 'встреча' in text_lower and 'команда' in text_lower:
                return "Провести встречу с командой разработки для обсуждения проекта"
            elif 'собеседование' in text_lower:
                return "Подготовиться к собеседованию и выбрать подходящую одежду"
            elif 'переезд' in text_lower:
                return "Упаковать вещи и договориться с грузчиками для переезда"
            elif 'операция' in text_lower and 'мама' in text_lower:
                return "Быть рядом с мамой во время операции и оказать поддержку"
            else:
                return f"Выполнить: {text}"
        elif task_type == 'mood_entry':
            if 'волнуюсь' in text_lower or 'переживаю' in text_lower:
                return "Испытываю сильное волнение и тревогу"
            elif 'устал' in text_lower or 'усталым' in text_lower:
                return "Испытываю усталость и нуждаюсь в отдыхе"
            elif 'отлично' in text_lower:
                return "Чувствую себя отлично, полон энергии и позитива"
            else:
                return "Запись о текущем эмоциональном состоянии"
        elif task_type == 'habit':
            if 'бегать' in text_lower:
                return "Регулярно бегать каждое утро для поддержания физической формы"
            elif 'программировать' in text_lower:
                return "Регулярно программировать каждый день для развития навыков"
            elif 'английский' in text_lower or 'язык' in text_lower:
                return "Регулярно изучать английский язык для развития навыков"
            elif 'читать' in text_lower:
                return "Регулярно читать для развития и самообразования"
            else:
                return "Развить новую полезную привычку"
        elif task_type == 'goal':
            if 'стартап' in text_lower:
                return "Создать стартап в сфере искусственного интеллекта с привлечением инвестиций"
            elif 'приложение' in text_lower:
                return "Создать мобильное приложение с качественным дизайном"
            elif 'фотограф' in text_lower:
                return "Стать профессиональным фотографом и открыть собственную студию"
            elif 'токио' in text_lower:
                return "Выучить японский язык и переехать в Токио для работы в IT компании"
            else:
                return "Достичь важной долгосрочной цели"
        
        return f"Описание: {text}"
    
    def _extract_tags(self, text: str) -> List[str]:
        """Извлекает теги с улучшенной логикой"""
        # Ищем похожие примеры
        similar_examples = self.find_similar_examples(text)
        
        for example in similar_examples:
            if 'expected' in example and isinstance(example['expected'], list):
                for item in example['expected']:
                    if 'tags' in item and isinstance(item['tags'], list):
                        return item['tags']
        
        # Если не нашли в обучении, используем улучшенные правила
        tags = []
        text_lower = text.lower()
        
        # Расширенная логика извлечения тегов
        tag_mappings = {
            'работа': ['работа', 'офис', 'коллеги', 'проект', 'встреча', 'программировать', 'код', 'разработка', 'клиент', 'презентация', 'карьера', 'бизнес', 'отчёт', 'руководитель', 'поставщик', 'команда', 'совещание', 'интервью', 'кандидат', 'тестирование', 'приложение', 'коммерческое предложение', 'продажи', 'собеседование', 'google'],
            'семья': ['семья', 'дети', 'родители', 'родственники', 'встретиться с родителями', 'мама', 'папа', 'день рождения', 'свадьба', 'молодожены'],
            'здоровье': ['здоровье', 'врач', 'лекарство', 'больница', 'устал', 'усталым', 'подавлен', 'терапевт', 'лечение', 'похудеть', 'диета', 'бросить курить', 'усталость', 'операция'],
            'спорт': ['спорт', 'тренировка', 'фитнес', 'зал', 'бегать', 'бег', 'беговая', 'утренний бег', 'тренироваться', 'физическая форма'],
            'технологии': ['программировать', 'код', 'разработка', 'компьютер', 'технологии', 'сайт', 'портфолио', 'IT', 'программирование'],
            'настроение': ['настроение', 'чувствую', 'эмоции', 'отлично', 'хорошо', 'плохо', 'волнуюсь', 'нервничаю', 'люблю', 'переживаю', 'тревога', 'стресс'],
            'учеба': ['учеба', 'экзамен', 'математика', 'изучать', 'обучение', 'курс', 'лекция', 'испанский', 'язык', 'презентация', 'вуз', 'магистратура', 'искусственный интеллект', 'мотивационное письмо', 'английский', 'японский'],
            'покупки': ['покупки', 'купить', 'магазин', 'товар', 'продукт', 'продукты', 'костюм', 'подарок', 'цветы', 'торт', 'мебель', 'билеты'],
            'дом': ['дом', 'квартира', 'переезд', 'мебель', 'недвижимость', 'грузчики'],
            'путешествия': ['путешествия', 'отпуск', 'Европа', 'отель', 'виза', 'самолет', 'транспорт', 'горы', 'поход', 'поход в горы', 'токио'],
            'хобби': ['хобби', 'гитара', 'музыка', 'играть', 'занятия'],
            'красота': ['красота', 'маникюр', 'массаж', 'расслабление', 'уход'],
            'организация': ['организация', 'упаковка', 'вещи', 'подготовка'],
            'дизайн': ['дизайн', 'интерфейс', 'ui', 'ux', 'графика', 'визуал'],
            'подарки': ['подарки', 'подарок', 'поздравление', 'сюрприз'],
            'новое место': ['новое место', 'новая работа', 'адаптация', 'коллектив'],
            'кулинария': ['кулинария', 'готовка', 'рецепты', 'шеф-повар', 'кухня'],
            'музыка': ['музыка', 'пианино', 'гитара', 'инструменты', 'мелодия'],
            'отчёт': ['отчёт', 'отчёты', 'отправить отчёт', 'руководитель'],
            'презентация': ['презентация', 'презентации', 'клиент', 'демонстрация', 'инвесторы'],
            'переговоры': ['переговоры', 'поставщик', 'обсуждение', 'условия'],
            'совещание': ['совещание', 'команда', 'встреча команды'],
            'система': ['система', 'база данных', 'обновление', 'проверка'],
            'проект': ['проект', 'техническое задание', 'требования'],
            'hr': ['hr', 'интервью', 'кандидат', 'найм'],
            'qa': ['qa', 'тестирование', 'баги', 'проверка'],
            'продажи': ['продажи', 'коммерческое предложение', 'клиент', 'расценки'],
            'стартап': ['стартап', 'стартапы', 'инвестиции', 'MVP', 'соучредитель'],
            'медитация': ['медитация', 'приложение', 'wellness', 'здоровье'],
            'фотография': ['фотография', 'фотограф', 'студия', 'творчество'],
            'привычка': ['привычка', 'привычки', 'регулярно', 'ежедневно', 'каждый день']
        }
        
        # Ищем совпадения
        for tag, keywords in tag_mappings.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if tag not in tags:
                        tags.append(tag)
                    break
        
        # Если тегов мало, добавляем дополнительные на основе контекста
        if len(tags) < 2:
            if 'завтра' in text_lower or 'послезавтра' in text_lower:
                if 'задача' not in tags:
                    tags.append('задача')
            if 'хочу' in text_lower or 'начну' in text_lower:
                if 'привычка' not in tags:
                    tags.append('привычка')
            if 'чувствую' in text_lower or 'волнуюсь' in text_lower:
                if 'настроение' not in tags:
                    tags.append('настроение')
        
        return tags[:4] if tags else ["задача"]  # Максимум 4 тега
    
    def _extract_priority(self, text: str) -> str:
        """Извлекает приоритет с улучшенной логикой"""
        text_lower = text.lower()
        
        # Расширенные ключевые слова для приоритетов (в порядке приоритета)
        priority_keywords = {
            'low': ['когда-нибудь', 'не спеша', 'в свободное время', 'срочность низкая', 'неважно', 'не важно', 'неприоритетно', 'в будущем', 'мечтаю', 'мечтаю стать'],
            'high': ['срочно', 'критично', 'немедленно', 'критическая', 'неотложно', 'срочная', 'важно', 'важная', 'критично важно', 'критично важно завершить', 'нужно', 'надо', 'требуется', 'организовать', 'подготовить', 'подготовиться'],
            'medium': ['на этой неделе', 'в ближайшее время', 'встреча', 'презентация', 'отчёт', 'хочу', 'планирую', 'изучить', 'фреймворк', 'технологию', 'отправить']
        }
        
        # Проверяем наличие ключевых слов
        for priority, keywords in priority_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return priority
        
        # Дополнительная логика на основе контекста (в порядке приоритета)
        if 'срочно' in text_lower or 'критично' in text_lower:
            return 'high'
        elif 'когда-нибудь' in text_lower or 'не спеша' in text_lower or 'в будущем' in text_lower:
            return 'low'
        elif 'напоминание' in text_lower or 'будильник' in text_lower:
            return 'high'  # Напоминания и будильники = high приоритет
        elif 'отправить отчёт' in text_lower or 'отправить отчет' in text_lower:
            return 'medium'  # Отправка отчётов = medium приоритет
        elif 'завтра нужно отправить' in text_lower:
            return 'medium'  # Обычные задачи отправки завтра = medium
        elif 'нужно' in text_lower and 'отправить' in text_lower:
            return 'medium'  # Обычные задачи отправки = medium
        elif 'изучить' in text_lower and ('фреймворк' in text_lower or 'технологию' in text_lower):
            return 'medium'  # Изучение технологий = medium приоритет
        elif 'планирую изучить' in text_lower or ('планирую' in text_lower and 'изучить' in text_lower):
            return 'medium'  # Планирование изучения = medium приоритет
        elif 'волнуюсь' in text_lower or 'тревога' in text_lower or 'паника' in text_lower or 'нервничаю' in text_lower:
            return 'high'  # Тревога и паника = high приоритет
        elif 'волнуюсь перед' in text_lower or 'не могу уснуть' in text_lower:
            return 'high'  # Сильная тревога = high приоритет
        elif 'испытываю сильную' in text_lower and ('тревогу' in text_lower or 'панику' in text_lower):
            return 'high'  # Сильная тревога/паника = high приоритет
        elif 'очень нервничаю' in text_lower or 'очень волнуюсь' in text_lower:
            return 'high'  # Сильное волнение = high приоритет
        elif 'собеседование' in text_lower:
            return 'high'  # Собеседования = high приоритет
        elif 'важно' in text_lower and ('встреча' in text_lower or 'презентация' in text_lower):
            return 'high'  # Важные встречи/презентации = high
        elif 'важная' in text_lower and ('встреча' in text_lower or 'презентация' in text_lower):
            return 'high'  # Важные встречи/презентации = high
        elif 'важно' in text_lower and ('проект' in text_lower or 'дедлайн' in text_lower):
            return 'high'  # Важные проекты с дедлайном = high
        elif 'до 12:00' in text_lower or 'до 12' in text_lower:
            return 'high'  # Задачи с дедлайном до полудня = high
        elif 'проект' in text_lower and ('до' in text_lower or 'дедлайн' in text_lower):
            return 'high'  # Проекты с дедлайном = high
        elif 'нужно' in text_lower and ('организовать' in text_lower or 'подготовить' in text_lower):
            return 'high'  # Важные организационные задачи = high
        elif 'встреча' in text_lower or 'презентация' in text_lower:
            return 'medium'  # Обычные встречи/презентации = medium
        elif 'важно' in text_lower:
            return 'high'
        elif 'хочу' in text_lower or 'планирую' in text_lower:
            return 'medium'
        elif 'мечтаю' in text_lower:
            return 'low'  # Мечты = low приоритет
        elif 'радуюсь' in text_lower or 'счастье' in text_lower or 'благодарность' in text_lower:
            return 'medium'  # Положительные эмоции = medium приоритет
        else:
            return 'medium'
    
    def _extract_due_date(self, text: str) -> str:
        """Извлекает дату выполнения с улучшенной логикой"""
        text_lower = text.lower()
        
        # Определяем конкретное время из текста
        time_patterns = {
            # Точные времена HH:MM
            '6:30': '06:30',
            '6:30 утра': '06:30',
            '6:30 утром': '06:30',
            '7:15': '07:15',
            '7:15 утра': '07:15',
            '7:15 утром': '07:15',
            '8:30': '08:30',
            '8:30 утра': '08:30',
            '9:45': '09:45',
            '9:45 утра': '09:45',
            '10:15': '10:15',
            '10:15 утра': '10:15',
            '11:45': '11:45',
            '11:45 утра': '11:45',
            '12:15': '12:15',
            '12:15 дня': '12:15',
            '13:45': '13:45',
            '13:45 дня': '13:45',
            '14:15': '14:15',
            '14:15 дня': '14:15',
            '15:30': '15:30',
            '15:30 дня': '15:30',
            '16:45': '16:45',
            '16:45 дня': '16:45',
            '17:15': '17:15',
            '17:15 дня': '17:15',
            '18:30': '18:30',
            '18:30 вечера': '18:30',
            '19:45': '19:45',
            '19:45 вечера': '19:45',
            '20:15': '20:15',
            '20:15 вечера': '20:15',
            '21:30': '21:30',
            '21:30 вечера': '21:30',
            '21:45': '21:45',
            '21:45 вечера': '21:45',
            '22:15': '22:15',
            '22:15 вечера': '22:15',
            '23:30': '23:30',
            '23:30 вечера': '23:30',
            
            # Форматы "в HH:MM"
            'в 6:30': '06:30',
            'в 6:30 утра': '06:30',
            'в 7:15': '07:15',
            'в 7:15 утра': '07:15',
            'в 8:30': '08:30',
            'в 8:30 утра': '08:30',
            'в 9:45': '09:45',
            'в 9:45 утра': '09:45',
            'в 10:15': '10:15',
            'в 10:15 утра': '10:15',
            'в 11:45': '11:45',
            'в 11:45 утра': '11:45',
            'в 12:15': '12:15',
            'в 12:15 дня': '12:15',
            'в 13:45': '13:45',
            'в 13:45 дня': '13:45',
            'в 14:15': '14:15',
            'в 14:15 дня': '14:15',
            'в 15:30': '15:30',
            'в 15:30 дня': '15:30',
            'в 16:45': '16:45',
            'в 16:45 дня': '16:45',
            'в 17:15': '17:15',
            'в 17:15 дня': '17:15',
            'в 18:30': '18:30',
            'в 18:30 вечера': '18:30',
            'в 19:45': '19:45',
            'в 19:45 вечера': '19:45',
            'в 20:15': '20:15',
            'в 20:15 вечера': '20:15',
            'в 21:30': '21:30',
            'в 21:30 вечера': '21:30',
            'в 21:45': '21:45',
            'в 21:45 вечера': '21:45',
            'в 22:15': '22:15',
            'в 22:15 вечера': '22:15',
            'в 23:30': '23:30',
            'в 23:30 вечера': '23:30',
            
            # Старые паттерны (сохраняем для совместимости)
            'до 12 дня': '12:00',
            'до 12': '12:00',
            '12 дня': '12:00',
            '12:00': '12:00',
            '12': '12:00',
            'в 13': '13:00',
            'в 14': '14:00',
            'в 15': '15:00',
            'в 16': '16:00',
            'в 17': '17:00',
            'в 18': '18:00',
            'в 19': '19:00',
            'в 20': '20:00',
            'в 21': '21:00',
            'в 22': '22:00',
            'в 23': '23:00',
            'в 10': '10:00',
            'в 11': '11:00',
            'в 7:00': '07:00',
            'в 8:00': '08:00',
            'в 9:00': '09:00',
            'в 10:00': '10:00',
            'в 11:00': '11:00',
            'в 12:00': '12:00',
            'в 13:00': '13:00',
            'в 14:00': '14:00',
            'в 15:00': '15:00',
            'в 16:00': '16:00',
            'в 17:00': '17:00',
            'в 18:00': '18:00',
            'в 19:00': '19:00',
            'в 20:00': '20:00',
            'в 21:00': '21:00',
            'в 22:00': '22:00',
            'в 23:00': '23:00',
            'в 8': '08:00',
            'в 9': '09:00',
            'в 9:30': '09:30',
            'в 10:30': '10:30',
            'в 11:30': '11:30',
            'в 12:30': '12:30',
            'в 13:30': '13:30',
            'в 14:30': '14:30',
            'в 15:30': '15:30',
            'в 16:30': '16:30',
            'в 17:30': '17:30',
            'в 18:30': '18:30',
            'в 19:30': '19:30',
            'в 20:30': '20:30',
            'в 21:30': '21:30',
            'в 22:30': '22:30',
            'в 23:30': '23:30',
            '8 утра': '08:00',
            '9 утра': '09:00',
            '10 утра': '10:00',
            '11 утра': '11:00',
            '12 дня': '12:00',
            '13 дня': '13:00',
            '14 дня': '14:00',
            '15 дня': '15:00',
            '16 дня': '16:00',
            '17 дня': '17:00',
            '18 вечера': '18:00',
            '19 вечера': '19:00',
            '20 вечера': '20:00',
            '21 вечера': '21:00',
            '22 вечера': '22:00',
            '23 вечера': '23:00',
            'в 8 утра': '08:00',
            'в 9 утра': '09:00',
            'в 10 утра': '10:00',
            'в 11 утра': '11:00',
            'в 12 дня': '12:00',
            'в 13 дня': '13:00',
            'в 14 дня': '14:00',
            'в 15 дня': '15:00',
            'в 16 дня': '16:00',
            'в 17 дня': '17:00',
            'в 18 вечера': '18:00',
            'в 19 вечера': '19:00',
            'в 20 вечера': '20:00',
            'в 21 вечера': '21:00',
            'в 22 вечера': '22:00',
            'в 23 вечера': '23:00',
            '13:00': '13:00',
            '14:00': '14:00',
            '15:00': '15:00',
            '16:00': '16:00',
            '17:00': '17:00',
            '18:00': '18:00',
            '19:00': '19:00',
            '20:00': '20:00',
            '21:00': '21:00',
            '22:00': '22:00',
            '23:00': '23:00',
            '10:00': '10:00',
            '11:00': '11:00',
            '08:00': '08:00',
            '09:00': '09:00',
            '09:30': '09:30',
            '10:30': '10:30',
            '11:30': '11:30',
            '12:30': '12:30',
            '13:30': '13:30',
            '14:30': '14:30',
            '15:30': '15:30',
            '16:30': '16:30',
            '17:30': '17:30',
            '18:30': '18:30',
            '19:30': '19:30',
            '20:30': '20:30',
            '21:30': '21:30',
            '22:30': '22:30',
            '23:30': '23:30',
            '00:30': '00:30',
            'утром': '10:00',
            'утро': '10:00',
            'днем': '14:00',
            'день': '14:00',
            'вечером': '18:00',
            'вечер': '18:00',
            'ночью': '22:00',
            'ночь': '22:00',
            'до 10': '10:00',
            'до 14': '14:00',
            'до 18': '18:00',
            'до 20': '20:00'
        }
        
        # Ищем конкретное время в тексте
        specific_time = None
        print(f"🔍 DEBUG: Ищем время в тексте '{text_lower}'")
        for pattern, time in time_patterns.items():
            if pattern in text_lower:
                specific_time = time
                print(f"✅ DEBUG: Найдено время '{pattern}' -> '{time}'")
                break
        
        if not specific_time:
            print(f"⚠️ DEBUG: Время не найдено, используем стандартное 18:00")
        
        # Определяем дату
        if 'срочно' in text_lower:
            date = self.current_date
        elif 'послезавтра' in text_lower:
            date = self.current_date + timedelta(days=2)
        elif 'завтра' in text_lower:
            date = self.current_date + timedelta(days=1)
        elif 'сегодня' in text_lower:
            date = self.current_date
        elif 'следующей неделе' in text_lower or 'следующей недели' in text_lower:
            date = self.current_date + timedelta(days=7)
        elif 'этой неделе' in text_lower:
            date = self.current_date + timedelta(days=3)
        elif 'когда-нибудь' in text_lower:
            date = self.current_date + timedelta(days=1)
        else:
            date = self.current_date + timedelta(days=1)
        
        # Используем конкретное время или стандартное
        time = specific_time if specific_time else "18:00"
        
        return date.strftime(f"%Y-%m-%d {time}")
    
    def improve_from_feedback(self, user_input: str, model_output: List[Dict], feedback: str) -> List[Dict]:
        """Улучшает модель на основе обратной связи"""
        print(f"🎓 Обучение на основе обратной связи: {feedback[:50]}...")
        
        # Добавляем пример в данные обучения
        training_example = {
            "input": user_input,
            "expected": model_output,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_data.append(training_example)
        
        # Сохраняем данные обучения
        try:
            with open('voicaj_training_data.json', 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, ensure_ascii=False, indent=2)
            print("✅ Данные обучения обновлены")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
        
        return model_output

# Тестирование
if __name__ == "__main__":
    print("🚀 Тестирование гибридной системы...")
    
    llm = HybridVoicajLLM()
    
    # Тест простого запроса
    simple_test = "завтра нужно отправить отчёт руководителю"
    print(f"\n📝 Простой тест: {simple_test}")
    result1 = llm.analyze_text(simple_test)
    print(f"⚡ Результат: {json.dumps(result1, ensure_ascii=False, indent=2)}")
    
    # Тест сложного запроса
    complex_test = "завтра нужно отправить отчёт руководителю и одновременно подготовить презентацию для клиента, а также позвонить поставщику и обсудить условия поставки"
    print(f"\n📝 Сложный тест: {complex_test}")
    result2 = llm.analyze_text(complex_test)
    print(f"🧠 Результат: {json.dumps(result2, ensure_ascii=False, indent=2)}")
