# Инструкция по запуску сервисов Preprocessing и ML

## Предварительные шаги

1. Убедитесь, что у вас установлен Python 3.7 или выше.
2. Установите необходимые библиотеки с помощью pip:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Убедитесь, что у вас есть файл модели resnet101.pth в той же директории, что и ml_service.py.

## Запуск сервисов

### 1. Сервис препроцессинга

Запустите сервис препроцессинга:

    ```bash
    uvicorn preprocessing_service:app --port 8000 --workers num_workers
    ```

### 2. Сервис с моделью

Создайте файл preprocessing_service.py и поместите в него соответствующий код.

Запустите сервис препроцессинга с 4 воркерами:

    ```bash
    uvicorn ml_service:app --port 8001 --workers num_workers
    ```
    
**num_workers** - количество одновременно запущенных процессов.
