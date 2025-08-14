## [2025-08-14] - Live guards (SQLite), drag-to-connect, edge labels, Qt6 backend fix
### Добавлено
- Движок Live Guard с поддержкой `expr:` и `db:` (SQLite) проверок
- UI-тестер guard-условий на вкладке Validators
- Drag & connect на графе (ПКМ: соединение узлов для создания переходов)
- Подписи на рёбрах графа: id, trigger, [guard], / actions
- Экспорт Mermaid/PlantUML теперь включает guard-условия и actions

### Изменено
- Backend Matplotlib для PyQt6: `backend_qtagg` вместо `backend_qt5agg`
- Автогенерация подписей рёбер для объединённых переходов между парой состояний

### Исправлено
- Ошибки Pylance: добавлены зависимости `matplotlib` и `toml` в `requirements.txt` 
## [2025-08-14] - Переход на tomllib, удаление строк в таблицах, современный UI
### Добавлено
- Кнопки удаления выбранных строк для таблиц: Variables, Sources, Actions, Timers, Global Transitions
- Переключатель темы (Light/Dark) и стиль `Fusion`

### Изменено
- Чтение TOML переведено на `tomllib` (Py3.11+)/`tomli` (fallback)
- Запись на `tomli-w`; пакет `toml` удалён из зависимостей

### Исправлено
- Единообразная индентация пробелами для новых блоков кода 
## [2025-08-14] - qt-material темы и автоподстановка
### Добавлено
- Интеграция `qt-material` ([PyPI](https://pypi.org/project/qt-material/)) с выбором темы и применением в рантайме
- Автоподстановка через `QCompleter`:
  - В переходах/глобальных переходах: источники (prefix `source.`), target-состояния, actions, заготовки для guard (`expr:`, `db:`)
  - В переменных: типы и persist (True/False)
  - В источниках: типы и шаблоны connection
  - В actions: kind/type
  - В таймерах: типы, события, флаги автостарта
- Completer-ы на карточке состояния (`on_enter`, `on_exit`) и поле `Initial State` 