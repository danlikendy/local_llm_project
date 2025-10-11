#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any

class VoicajLLM:
    """Voicaj LLM с системой обучения"""
    
    def __init__(self):
        self.training_data = []
        self.current_date = datetime.now()
        self.load_training_data()
        
        # Паттерны для определения типов задач
        self.patterns = {
            'task': [
                r'задача|завтра|нужно|должен|обязан|поставь|сделать|выполнить',
                r'встретиться|встреча|свидание|свиданье|встречаться',
                r'купить|покупка|магазин|товар|продукт|приобрести',
                r'врач|больница|лекарство|лечение|консультация|прием',
                r'экзамен|экзамены|сдать|подготовиться|готовиться',
                r'переезд|переезжать|упаковать|грузчики',
                r'оформить|забронировать|записаться|договориться',
                r'свадьба|день рождения|праздник|поздравить',
                r'презентация|выступление|доклад|отчет'
            ],
            'mood_entry': [
                r'чувствую|настроение|эмоции|ощущаю|кажется|состояние',
                r'устал|усталым|устала|усталость|измотан|устало',
                r'отлично|хорошо|плохо|грустно|весело|счастлив|радостно',
                r'подавлен|депрессия|стресс|тревожно|волнуюсь|переживаю',
                r'нервничаю|волнуюсь|боюсь|тревожусь|переживаю',
                r'люблю|обожаю|нравится|восхищаюсь',
                r'неуверен|сомневаюсь|боюсь|тревожусь'
            ],
            'habit': [
                r'привычка|привычки|регулярно|каждый день|ежедневно|начать',
                r'бегать|бег|тренировка|спорт|фитнес|зал|тренироваться',
                r'программировать|код|разработка|программа|изучать',
                r'читать|книга|лекция|курс|обучение|изучать',
                r'играть на гитаре|музыка|заниматься музыкой',
                r'есть овощи|диета|питание|здоровое питание',
                r'жвачки|никотиновые|бросить курить|отказ от курения'
            ],
            'health': [
                r'здоровье|здоровый|здоровая|здоровое|самочувствие',
                r'болезнь|лечение|терапия|анализы|диагноз|симптомы',
                r'боль|болит|таблетки|укол|операция|реабилитация',
                r'похудеть|сбросить вес|диета|калории',
                r'бросить курить|отказ от курения|никотин'
            ],
            'workout': [
                r'тренировка|тренироваться|спорт|фитнес|зал|стадион',
                r'бег|бегать|йога|плавание|велосипед|футбол|баскетбол',
                r'массаж|сауна|бассейн|тренажер|гантели|кардио'
            ],
            'goal': [
                r'цель|цели|планирую|хочу|мечтаю|стремлюсь|желаю',
                r'достичь|достигнуть|получить|заработать|выучить',
                r'стать|быть|работать|жить|путешествовать',
                r'открыть бизнес|создать|запустить|развить',
                r'похудеть|сбросить вес|улучшить здоровье',
                r'научиться|освоить|изучить|выучить'
            ]
        }
        
        # База тегов
        self.tags_keywords = {
            'работа': ['работа', 'офис', 'коллеги', 'проект', 'встреча', 'программировать', 'код', 'разработка', 'клиент', 'презентация', 'карьера', 'бизнес', 'отчёт', 'руководитель', 'поставщик', 'команда', 'совещание', 'интервью', 'кандидат', 'тестирование', 'приложение', 'коммерческое предложение', 'продажи'],
            'семья': ['семья', 'дети', 'родители', 'родственники', 'встретиться с родителями', 'мама', 'папа', 'день рождения', 'свадьба', 'молодожены'],
            'здоровье': ['здоровье', 'врач', 'лекарство', 'больница', 'устал', 'усталым', 'подавлен', 'терапевт', 'лечение', 'похудеть', 'диета', 'бросить курить'],
            'спорт': ['спорт', 'тренировка', 'фитнес', 'зал', 'бегать', 'бег', 'беговая', 'утренний бег', 'тренироваться', 'физическая форма'],
            'технологии': ['программировать', 'код', 'разработка', 'компьютер', 'технологии', 'сайт', 'портфолио'],
            'программирование': ['код', 'программирование', 'написание кода', 'разработка', 'программировать'],
            'настроение': ['настроение', 'чувствую', 'эмоции', 'отлично', 'хорошо', 'плохо', 'волнуюсь', 'нервничаю', 'люблю', 'переживаю'],
            'учеба': ['учеба', 'экзамен', 'математика', 'изучать', 'обучение', 'курс', 'лекция', 'испанский', 'язык', 'презентация', 'вуз'],
            'покупки': ['покупки', 'купить', 'магазин', 'товар', 'продукт', 'продукты', 'костюм', 'подарок', 'цветы', 'торт', 'мебель', 'билеты'],
            'дом': ['дом', 'квартира', 'переезд', 'мебель', 'недвижимость', 'грузчики'],
            'путешествия': ['путешествия', 'отпуск', 'Европа', 'отель', 'виза', 'самолет', 'транспорт', 'горы', 'поход', 'поход в горы'],
            'хобби': ['хобби', 'гитара', 'музыка', 'играть', 'занятия'],
            'красота': ['красота', 'маникюр', 'массаж', 'расслабление', 'уход'],
            'право': ['право', 'паспорт', 'юрист', 'договор', 'консультация', 'документы'],
            'события': ['события', 'свадьба', 'день рождения', 'праздник', 'поздравление'],
            'привычки': ['привычки', 'привычка', 'регулярно', 'ежедневно', 'каждый день'],
            'цели': ['цели', 'цель', 'планирую', 'хочу', 'мечтаю', 'стремлюсь'],
            'друзья': ['друзья', 'друг', 'свадьба', 'молодожены', 'поздравление', 'поговорить с друзьями', 'разговор с друзьями'],
            'финансы': ['финансы', 'депозит', 'банк', 'деньги', 'накопления'],
            'организация': ['организация', 'упаковка', 'вещи', 'подготовка'],
            'дизайн': ['дизайн', 'интерфейс', 'ui', 'ux', 'графика', 'визуал'],
            'подарки': ['подарки', 'подарок', 'поздравление', 'сюрприз'],
            'новое место': ['новое место', 'новая работа', 'адаптация', 'коллектив'],
            'кулинария': ['кулинария', 'готовка', 'рецепты', 'шеф-повар', 'кухня'],
            'музыка': ['музыка', 'пианино', 'гитара', 'инструменты', 'мелодия'],
            'отчёт': ['отчёт', 'отчёты', 'отправить отчёт', 'руководитель'],
            'презентация': ['презентация', 'презентации', 'клиент', 'демонстрация'],
            'переговоры': ['переговоры', 'поставщик', 'обсуждение', 'условия'],
            'совещание': ['совещание', 'команда', 'встреча команды'],
            'система': ['система', 'база данных', 'обновление', 'проверка'],
            'проект': ['проект', 'техническое задание', 'требования'],
            'hr': ['hr', 'интервью', 'кандидат', 'найм'],
            'qa': ['qa', 'тестирование', 'баги', 'проверка'],
            'продажи': ['продажи', 'коммерческое предложение', 'клиент', 'расценки']
        }
        
        # Ключевые слова для приоритетов
        self.priority_keywords = {
            'high': ['срочно', 'важно', 'критично', 'немедленно', 'сегодня', 'завтра'],
            'medium': ['на этой неделе', 'в ближайшее время'],
            'low': ['когда-нибудь', 'не спеша', 'в свободное время', 'срочность низкая', 'неважно', 'не важно']
        }

    def load_training_data(self):
        """Загружает данные обучения"""
        try:
            # Безопасная загрузка без print
            with open('voicaj_training_data.json', 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
        except Exception:
            self.training_data = []
            
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

    def improve_from_feedback(self, user_input: str, original_output: List[Dict], feedback: str):
        """Улучшает модель на основе обратной связи"""
        # Анализируем обратную связь
        improvements = self._analyze_feedback(feedback)
        
        # Создаем улучшенный вывод
        improved_output = self._apply_improvements(original_output, improvements)
        
        # Добавляем в базу обучения
        self._add_training_example(user_input, improved_output, feedback)
        
        return improved_output
        
    def _analyze_feedback(self, feedback: str) -> Dict[str, List[str]]:
        """Анализирует обратную связь и определяет области для улучшения"""
        improvements = {
            'description_improvements': [],
            'date_improvements': [],
            'tag_improvements': [],
            'title_improvements': [],
            'priority_improvements': []
        }
        
        feedback_lower = feedback.lower()
        
        if 'описание' in feedback_lower and ('общее' in feedback_lower or 'мало' in feedback_lower):
            improvements['description_improvements'].append('make_more_specific')
            
        if 'дата' in feedback_lower and ('неправильная' in feedback_lower or 'неверная' in feedback_lower):
            improvements['date_improvements'].append('fix_date')
            
        if 'тег' in feedback_lower and ('нет' in feedback_lower or 'мало' in feedback_lower):
            improvements['tag_improvements'].append('add_more_tags')
            
        if 'заголовок' in feedback_lower and ('неточный' in feedback_lower or 'общий' in feedback_lower):
            improvements['title_improvements'].append('make_specific_title')
            
        return improvements
        
    def _apply_improvements(self, original_output: List[Dict], improvements: Dict[str, List[str]]) -> List[Dict]:
        """Применяет улучшения к исходному выводу"""
        improved_output = original_output.copy()
        
        for item in improved_output:
            # Улучшаем описания
            if 'make_more_specific' in improvements['description_improvements']:
                item['description'] = self._make_description_more_specific(item['description'], item['type'])
                
            # Исправляем даты
            if 'fix_date' in improvements['date_improvements'] and 'dueDate' in item:
                item['dueDate'] = self._fix_date(item['dueDate'])
                
            # Добавляем теги
            if 'add_more_tags' in improvements['tag_improvements']:
                item['tags'] = self._add_more_tags(item['tags'], item['type'])
                
            # Улучшаем заголовки
            if 'make_specific_title' in improvements['title_improvements']:
                item['title'] = self._make_title_more_specific(item['title'], item['type'])
                
        return improved_output
        
    def _make_description_more_specific(self, description: str, task_type: str) -> str:
        """Делает описание более конкретным"""
        if task_type == 'task':
            if 'встреча' in description.lower():
                return "Провести важную встречу для обсуждения проекта и принятия решений"
            elif 'задача' in description.lower():
                return "Выполнить поставленную задачу в соответствии с требованиями"
        return description
        
    def _fix_date(self, date_str: str) -> str:
        """Исправляет дату"""
        # Простая логика исправления даты
        if '2025-10-08' in date_str:
            return date_str.replace('2025-10-08', '2025-10-07')
        return date_str
        
    def _add_more_tags(self, tags: List[str], task_type: str) -> List[str]:
        """Добавляет больше тегов"""
        if len(tags) < 2:
            if task_type == 'task':
                tags.extend(['работа', 'важно'])
            elif task_type == 'mood_entry':
                tags.extend(['настроение', 'эмоции'])
        return tags[:3]
        
    def _make_title_more_specific(self, title: str, task_type: str) -> str:
        """Делает заголовок более конкретным"""
        if title == 'Задача' and task_type == 'task':
            return 'Важная задача'
        elif title == 'Запись настроения' and task_type == 'mood_entry':
            return 'Запись эмоционального состояния'
        return title
        
    def _add_training_example(self, user_input: str, improved_output: List[Dict], feedback: str):
        """Добавляет пример в базу обучения"""
        example = {
            'input': user_input,
            'expected': improved_output,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data.append(example)
        
        # Сохраняем в файл
        try:
            with open('voicaj_training_data.json', 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Игнорируем ошибки записи

    def analyze_text(self, text: str) -> List[Dict[str, Any]]:
        """Анализирует текст"""
        # Всегда создаем новый анализ, а не возвращаем старые примеры
        detected_types = self._detect_types(text.lower())
        
        if len(detected_types) == 0:
            detected_types = ['task']
            
        # Создаем объекты
        results = []
        for task_type in detected_types:
            result = self._create_object(text, task_type)
            if result:
                results.append(result)
        
        # Ищем похожие примеры для улучшения результата
        similar_examples = self.find_similar_examples(text)
        if similar_examples:
            # Улучшаем результат на основе обучения
            for i, result in enumerate(results):
                results[i] = self._improve_with_training_data(result, similar_examples, text)
                    
        return results
    
    def _improve_with_training_data(self, result: Dict, similar_examples: List[Dict], text: str) -> Dict:
        """Улучшает результат на основе данных обучения"""
        if not similar_examples:
            return result
            
        # Берем первый похожий пример
        example = similar_examples[0]
        expected = example.get('expected', [])
        
        if expected:
            # Ищем подходящий объект в ожидаемом результате
            for exp_obj in expected:
                if exp_obj.get('type') == result.get('type'):
                    # Обновляем поля на основе обучения
                    result['title'] = exp_obj.get('title', result['title'])
                    result['description'] = exp_obj.get('description', result['description'])
                    result['tags'] = exp_obj.get('tags', result['tags'])
                    result['priority'] = exp_obj.get('priority', result['priority'])
                    
                    # Обновляем дату если есть, но адаптируем под текущий запрос
                    if 'dueDate' in exp_obj:
                        result['dueDate'] = self._adapt_date_for_request(exp_obj['dueDate'], text)
                    elif 'dueDate' in result:
                        # Адаптируем дату под текущий запрос
                        result['dueDate'] = self._adapt_date_for_request(result['dueDate'], text)
                    
                    # Обновляем частоту для привычек
                    if 'frequency' in exp_obj:
                        result['frequency'] = exp_obj['frequency']
                    
                    break
        
        return result
    
    def _adapt_date_for_request(self, date_str: str, text: str) -> str:
        """Адаптирует дату под текущий запрос"""
        text_lower = text.lower()
        
        # Определяем дату из текста запроса
        if 'послезавтра' in text_lower:
            day_after = self.current_date + timedelta(days=2)
            return day_after.strftime("%Y-%m-%d 18:00")
        elif 'завтра' in text_lower:
            tomorrow = self.current_date + timedelta(days=1)
            return tomorrow.strftime("%Y-%m-%d 18:00")
        elif 'сегодня' in text_lower:
            return self.current_date.strftime("%Y-%m-%d 18:00")
        elif 'следующей неделе' in text_lower or 'следующей недели' in text_lower:
            next_week = self.current_date + timedelta(days=7)
            return next_week.strftime("%Y-%m-%d 18:00")
        elif 'этой неделе' in text_lower:
            this_week = self.current_date + timedelta(days=3)
            return this_week.strftime("%Y-%m-%d 18:00")
        else:
            return date_str
        
    def _detect_types(self, text: str) -> List[str]:
        """Определяет типы задач в тексте"""
        detected = set()
        
        for task_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    detected.add(task_type)
                    break  # Переходим к следующему типу после первого совпадения
                    
        return list(detected)
        
    def _create_object(self, text: str, task_type: str) -> Dict[str, Any]:
        """Создает объект"""
        obj = {
            "title": self._extract_title(text, task_type),
            "type": task_type,
            "description": self._extract_description(text, task_type),
            "priority": self._extract_priority(text),
            "tags": self._extract_tags(text),
            "timestamp": datetime.now().isoformat()
        }
        
        # Добавляем специфичные поля
        if task_type == "task":
            obj["dueDate"] = self._extract_due_date(text)
        elif task_type == "habit":
            obj["frequency"] = self._extract_frequency(text)
        elif task_type == "workout":
            obj["duration"] = self._extract_duration(text)
            
        return obj
        
    def _extract_title(self, text: str, task_type: str) -> str:
        """Извлекает заголовок с использованием данных обучения"""
        # Ищем похожие примеры
        similar_examples = self.find_similar_examples(text)
        
        for example in similar_examples:
            if 'expected' in example and isinstance(example['expected'], list):
                for item in example['expected']:
                    if item.get('type') == task_type and 'title' in item:
                        return item['title']
        
        # Если не нашли в обучении, используем базовые правила
        text_lower = text.lower()
        
        if task_type == 'task':
            if 'отчёт' in text_lower and 'руководитель' in text_lower:
                return "Отправка отчёта руководителю"
            elif 'презентация' in text_lower and 'клиент' in text_lower:
                return "Подготовка презентации для клиента"
            elif 'позвонить' in text_lower and 'поставщик' in text_lower:
                return "Звонок поставщику"
            elif 'совещание' in text_lower and 'команда' in text_lower:
                return "Совещание с командой"
            elif 'база данных' in text_lower or 'система' in text_lower:
                return "Обновление базы данных"
            elif 'техническое задание' in text_lower:
                return "Написание технического задания"
            elif 'интервью' in text_lower and 'кандидат' in text_lower:
                return "Интервью с кандидатом"
            elif 'тестирование' in text_lower and 'приложение' in text_lower:
                return "Тестирование новой функциональности"
            elif 'коммерческое предложение' in text_lower:
                return "Отправка коммерческого предложения"
            elif 'презентация' in text_lower and 'вуз' in text_lower:
                return "Презентация по вузу"
            elif 'код' in text_lower and 'работа' in text_lower:
                return "Написание кода по работе"
            elif 'встретиться с родителями' in text_lower:
                return "Встреча с родителями"
            elif 'врач' in text_lower:
                return "Визит к врачу"
            elif 'программировать' in text_lower:
                return "Изучение программирования"
            elif 'продукт' in text_lower or 'магазин' in text_lower or 'купить' in text_lower:
                return "Покупка продуктов"
            elif 'сходить' in text_lower and ('продукт' in text_lower or 'магазин' in text_lower):
                return "Поход за продуктами"
            else:
                return "Задача"
                
        elif task_type == 'mood_entry':
            if 'отлично' in text_lower:
                return "Отличное настроение"
            elif 'устал' in text_lower:
                return "Усталость"
            else:
                return "Запись настроения"
                
        elif task_type == 'habit':
            if 'программировать' in text_lower:
                return "Изучение программирования"
            elif 'бегать' in text_lower:
                return "Утренний бег"
            else:
                return "Новая привычка"
                
        return "Запись"
        
    def _extract_description(self, text: str, task_type: str) -> str:
        """Извлекает описание с использованием данных обучения"""
        # Ищем похожие примеры
        similar_examples = self.find_similar_examples(text)
        
        for example in similar_examples:
            if 'expected' in example and isinstance(example['expected'], list):
                for item in example['expected']:
                    if item.get('type') == task_type and 'description' in item:
                        return item['description']
        
        # Если не нашли в обучении, используем базовые правила
        text_lower = text.lower()
        
        if task_type == 'task':
            if 'отчёт' in text_lower and 'руководитель' in text_lower:
                return "Подготовить и отправить отчёт руководителю о выполненных задачах до конца рабочего дня"
            elif 'презентация' in text_lower and 'клиент' in text_lower:
                return "Подготовить презентацию для встречи с клиентом с детальным описанием проекта"
            elif 'позвонить' in text_lower and 'поставщик' in text_lower:
                return "Позвонить поставщику для обсуждения условий поставки и сроков доставки"
            elif 'совещание' in text_lower and 'команда' in text_lower:
                return "Провести совещание с командой для обсуждения текущих проектов и планов"
            elif 'база данных' in text_lower or 'система' in text_lower:
                return "Обновить базу данных и проверить работоспособность системы после обновления"
            elif 'техническое задание' in text_lower:
                return "Написать детальное техническое задание для нового проекта с описанием требований и сроков"
            elif 'интервью' in text_lower and 'кандидат' in text_lower:
                return "Провести техническое интервью с кандидатом на должность разработчика"
            elif 'тестирование' in text_lower and 'приложение' in text_lower:
                return "Провести полное тестирование новой функциональности приложения и составить отчёт о найденных багах"
            elif 'коммерческое предложение' in text_lower:
                return "Подготовить и отправить коммерческое предложение клиенту с расценками и условиями сотрудничества"
            elif 'презентация' in text_lower and 'вуз' in text_lower:
                return "Подготовить презентацию по вузу для демонстрации результатов обучения"
            elif 'код' in text_lower and 'работа' in text_lower:
                return "Выполнить задачу по написанию кода для работы"
            elif 'встретиться с родителями' in text_lower:
                return "Встретиться с родителями для общения и поддержания семейных связей"
            elif 'врач' in text_lower:
                return "Записаться на прием к врачу для консультации и обследования"
            elif 'программировать' in text_lower:
                return "Начать изучение программирования для развития профессиональных навыков"
            elif 'продукт' in text_lower or 'магазин' in text_lower or 'купить' in text_lower:
                return "Сходить в магазин за продуктами для дома"
            elif 'сходить' in text_lower and ('продукт' in text_lower or 'магазин' in text_lower):
                return "Сходить за продуктами до конца дня"
            else:
                return "Выполнить поставленную задачу"
                
        elif task_type == 'mood_entry':
            if 'отлично' in text_lower:
                return "Чувствую себя отлично, полон энергии и позитива"
            elif 'устал' in text_lower:
                return "Испытываю усталость и нуждаюсь в отдыхе"
            else:
                return "Запись о текущем эмоциональном состоянии"
                
        elif task_type == 'habit':
            if 'программировать' in text_lower:
                return "Регулярно изучать программирование для профессионального развития"
            elif 'бегать' in text_lower:
                return "Ежедневно бегать по утрам для поддержания физической формы"
            else:
                return "Развить новую полезную привычку"
                
        return "Описание задачи"
        
    def _extract_due_date(self, text: str) -> str:
        """Извлекает дату выполнения с использованием данных обучения"""
        text_lower = text.lower()
        
        # Сначала определяем дату из текста
        if 'послезавтра' in text_lower:
            day_after = self.current_date + timedelta(days=2)
            return day_after.strftime("%Y-%m-%d 18:00")
        elif 'завтра' in text_lower:
            tomorrow = self.current_date + timedelta(days=1)
            return tomorrow.strftime("%Y-%m-%d 18:00")
        elif 'сегодня' in text_lower:
            return self.current_date.strftime("%Y-%m-%d 18:00")
        elif 'следующей неделе' in text_lower or 'следующей недели' in text_lower:
            next_week = self.current_date + timedelta(days=7)
            return next_week.strftime("%Y-%m-%d 18:00")
        elif 'этой неделе' in text_lower:
            this_week = self.current_date + timedelta(days=3)
            return this_week.strftime("%Y-%m-%d 18:00")
        else:
            # По умолчанию завтра
            tomorrow = self.current_date + timedelta(days=1)
            return tomorrow.strftime("%Y-%m-%d 18:00")
            
    def _extract_tags(self, text: str) -> List[str]:
        """Извлекает теги с использованием данных обучения"""
        # Ищем похожие примеры
        similar_examples = self.find_similar_examples(text)
        
        for example in similar_examples:
            if 'expected' in example and isinstance(example['expected'], list):
                for item in example['expected']:
                    if 'tags' in item and isinstance(item['tags'], list):
                        return item['tags']
        
        # Если не нашли в обучении, используем базовые правила
        tags = []
        text_lower = text.lower()
        
        for tag, keywords in self.tags_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if tag not in tags:
                        tags.append(tag)
                    break
                    
        return tags[:3]
        
    def _extract_priority(self, text: str) -> str:
        """Извлекает приоритет с использованием данных обучения"""
        # Ищем похожие примеры
        similar_examples = self.find_similar_examples(text)
        
        for example in similar_examples:
            if 'expected' in example and isinstance(example['expected'], list):
                for item in example['expected']:
                    if 'priority' in item:
                        return item['priority']
        
        # Если не нашли в обучении, используем базовые правила
        text_lower = text.lower()
        
        for priority, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return priority
                    
        return "medium"
        
    def _extract_frequency(self, text: str) -> str:
        """Извлекает частоту для привычек"""
        text_lower = text.lower()
        
        if 'каждый день' in text_lower or 'ежедневно' in text_lower:
            return "daily"
        elif 'каждую неделю' in text_lower:
            return "weekly"
        else:
            return "daily"
            
    def _extract_duration(self, text: str) -> str:
        """Извлекает продолжительность для тренировок"""
        return "30 минут"