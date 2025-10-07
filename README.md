# Voicaj LLM API Сервер

Локальный AI-ассистент со структурированным JSON выводом на основе Voicaj LLM Schema. Работает на вашем компьютере без подключения к интернету и предоставляет интеллектуальную категоризацию пользовательского ввода для интеграции с iOS/mobile приложениями.

## Особенности

- **Полная приватность** - Все данные остаются на вашем компьютере
- **Структурированный JSON вывод** - Интеллектуальная категоризация с использованием Voicaj LLM Schema
- **Память разговоров** - AI запоминает предыдущие сообщения в сессии
- **Сетевой доступ** - Подключение с любого устройства в локальной сети
- **Быстрая работа** - Модель работает на вашем оборудовании
- **Готов для iOS** - Идеально подходит для интеграции с мобильными приложениями

## Быстрый запуск

### 1. Запуск сервера
```bash
# Запустите скрипт
.\start_server.bat
```

### 2. Подключение
- **Локально**: http://localhost:5000
- **Из сети**: http://[ВАШ_IP]:5000

### 3. Получение IP адреса
```bash
powershell -ExecutionPolicy Bypass -File get_ip.ps1
```

## Системные требования

- Windows 10 / 11
- Python 3.8+
- Минимум 8 ГБ RAM (рекомендуется 16 ГБ)
- Свободное место: 10 ГБ для модели

## Установка

### Ollama
```bash
winget install Ollama.Ollama
```

### Python зависимости
```bash
pip install flask flask-cors requests python-dotenv
```

### AI Модель
```bash
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.1:8b
```

## Структура проекта

```
D:\local_ollama_project\
├── app.py                 # Основное Flask приложение с Voicaj Schema
├── templates/
│   └── index.html         # Веб-интерфейс
├── chat_history.db        # База данных истории разговоров
├── start_server.bat       # Скрипт запуска
<<<<<<< HEAD
├── get_ip.ps1            # Получение IP адресов
├── requirements.txt       # Python зависимости
└── .gitignore            # Git ignore файл
=======
├── get_ip.ps1             # Получение IP адресов
└── requirements.txt       # Python зависимости
>>>>>>> c34b88ec10cefd61ef3fd2b8a827040f83a93070
```

## Конфигурация

### Изменение модели
В файле `app.py`:
```python
MODEL_NAME = "llama3.1:8b"  # Измените на вашу модель
```

### Доступные модели
- `llama3.1:8b` - Рекомендуемая (4.9 ГБ)
- `llama3.2:3b` - Быстрая (2.0 ГБ)
- `qwen2.5:7b` - Альтернативная
- `mistral:7b` - Другая опция

## Voicaj LLM Schema

API возвращает структурированный JSON на основе анализа пользовательского ввода:

### Поддерживаемые типы:
- **task** - Задачи, дедлайны, напоминания
- **diary_entry** - Личные заметки, размышления
- **habit** - Повторяющиеся действия, цели
- **health** - Показатели здоровья, сон, шаги
- **workout** - Физические активности, упражнения
- **meal** - Логирование еды, приемы пищи
- **goal** - Долгосрочные цели
- **advice** - Коучинг, запросы помощи
- **study_note** - Обучение, учебные материалы
- **time_log** - Учет времени
- **shared_task** - Совместные задачи
- **focus_session** - Pomodoro сессии
- **mood_entry** - Эмоциональное состояние
- **expense** - Финансовые транзакции
- **travel_plan** - Планирование путешествий

### Формат ответа API:
```json
{
  "response": {
    "type": "task",
    "title": "Завершить отчет по проекту",
    "description": "Завершить и отправить отчет по проекту",
    "priority": "high",
    "tags": ["работа", "дедлайн"],
    "dueDate": "2025-01-10 17:00",
    "address": null
  },
  "session_id": "uuid-string",
  "type": "structured_json"
}
```

## API Эндпоинты

### Основной чат эндпоинт
```http
POST /api/chat
Content-Type: application/json

{
  "message": "Мне нужно завершить отчет по проекту к пятнице"
}
```

### Другие эндпоинты
- **Получить историю**: `GET /api/history`
- **Очистить историю**: `POST /api/clear`
- **Получить модели**: `GET /api/models`

## Интеграция с iOS

### Пример Swift:
```swift
struct VoicajResponse: Codable {
    let response: VoicajData
    let sessionId: String
    let type: String
    
    enum CodingKeys: String, CodingKey {
        case response
        case sessionId = "session_id"
        case type
    }
}

struct VoicajData: Codable {
    let type: String
    let title: String
    let description: String?
    let priority: String?
    let tags: [String]?
    let dueDate: String?
    let startDate: String?
    let timestamp: String?
}

func sendMessage(_ message: String) async throws -> VoicajResponse {
    let url = URL(string: "http://192.168.1.28:5000/api/chat")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let body = ["message": message]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    return try JSONDecoder().decode(VoicajResponse.self, from: data)
}
```

## Сетевой доступ

### Подключение в локальной сети
- Устройства должны быть в одной Wi-Fi сети
- Брандмауэр должен разрешать подключения на порт 5000

### Настройка брандмауэра
```bash
# PowerShell от администратора
New-NetFirewallRule -DisplayName "Voicaj AI Assistant" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

### Подключение с других устройств
1. Получите IP сервера: `get_ip.ps1`
2. Откройте браузер на другом устройстве
3. Перейдите по адресу: `http://[IP_СЕРВЕРА]:5000`

## Использование

### Локальное использование
1. **Запустите сервер**: `.\start_server.bat`
2. **Откройте браузер**: http://localhost:5000
3. **Начните общение**: Введите сообщение в поле ввода

### Использование по сети
1. Получите IP сервера: `get_ip.ps1`
2. Откройте браузер на другом устройстве
3. Перейдите по адресу: `http://[IP_СЕРВЕРА]:5000`

## База данных

История разговоров сохраняется в SQLite базе `chat_history.db`:
- Каждая сессия имеет уникальный ID
- Сохраняются сообщения пользователя и ответы AI
- История доступна в рамках сессии

## Безопасность

- Все данные обрабатываются локально
- Никакая информация не передается в интернет
- История разговоров хранится только на вашем компьютере
- Доступ только через локальную сеть

## Решение проблем

### Ollama не запускается
```bash
# Проверьте установку
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version

# Перезапустите сервис
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

### Модель не загружается
```bash
# Проверьте доступные модели
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list

# Загрузите модель заново
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.1:8b
```

### Не подключается с других устройств
1. Проверьте IP адрес: `get_ip.ps1`
2. Убедитесь что устройства в одной сети
3. Проверьте настройки брандмауэра
4. Попробуйте отключить антивирус временно

### Медленная работа
- Закройте другие программы
- Увеличьте объем RAM
- Используйте модель меньшего размера (llama3.2:3b)

## Поддержка

При возникновении проблем:
1. Проверьте логи в консоли
2. Убедитесь что все компоненты установлены
3. Проверьте доступность портов 11434 и 5000

## Обновление

### Обновление модели
```bash
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.1:8b
```

### Обновление приложения
- Замените файлы `app.py` и `templates/index.html`
- Перезапустите сервер
