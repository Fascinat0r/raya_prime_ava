import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Построение матрицы ошибок
def plot_confusion_matrix(y_test, y_pred_labels):
    """Построение и отображение матрицы ошибок."""
    cm = confusion_matrix(y_test, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Разные голоса', 'Голос Райи'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Матрица ошибок для модели распознавания голоса")
    plt.show()


# Добавление EarlyStopping для контроля за переобучением
def get_early_stopping_callback(patience=3):
    """
    Создаёт обратный вызов EarlyStopping для контроля за переобучением.

    Args:
        patience (int): Количество эпох без улучшения для остановки обучения.

    Returns:
        tf.keras.callbacks.EarlyStopping: Обратный вызов.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
