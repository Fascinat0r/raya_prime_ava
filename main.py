import time

from config import Config
from logger import get_logger
from preprocessing.full_processing_pipeline import run_parallel_processing
from train.cross_entropy.cross_entropy_main import cross_entropy_train
from train.split_data import split_melspec_data

logger = get_logger(__name__)


def main():
    # Загружаем конфигурацию
    config = Config()

    start_time = time.time()

    # Запускаем обработку данных
    run_parallel_processing(config)

    # Разделяем данные на тренировочные и тестовые
    split_melspec_data(config.METADATA_PATH)

    #  Запускаем обучение модели с использованием кросс-энтропии
    cross_entropy_train(config)

    logger.info(f"Общее время выполнения: {time.time() - start_time:.2f} секунд.")


if __name__ == "__main__":
    main()
