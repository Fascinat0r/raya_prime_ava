import logging
import os


# Создание конфигурации логирования
def setup_logger(log_file="logs.log") -> logging.Logger:
    """
    Создает и настраивает глобальный логгер для использования в проекте.
    :param log_file: Имя файла для сохранения логов.
    :return: Настроенный логгер.
    """
    # Убедимся, что директория для логов существует
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Настройка логгера
    _logger = logging.getLogger("data_preparation_logger")
    _logger.setLevel(logging.DEBUG)  # Устанавливаем уровень логирования на DEBUG для детальной информации

    # Создание формата логирования
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Консольный обработчик логов (для вывода на экран)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # На экран выводятся только информационные и выше уровни
    console_handler.setFormatter(formatter)

    # Файловый обработчик логов
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # В файл записываются все уровни логов
    file_handler.setFormatter(formatter)

    # Добавление обработчиков в логгер
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)

    return _logger


# Глобальный логгер для использования
logger = setup_logger("logs.log")
