import logging
import os

# Папка для хранения логов
LOG_FOLDER = "logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)


# Настройки логгера
def get_logger(logger_name):
    """
    Возвращает настроенный логгер с именем `logger_name`.

    :param logger_name: Имя логгера.
    :return: Объект логгера.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Создание обработчика для вывода в файл
    file_handler = logging.FileHandler(os.path.join(LOG_FOLDER, f"{logger_name}.log"), mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Создание обработчика для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Формат логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Добавление обработчиков к логгеру
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
