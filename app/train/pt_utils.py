# app/train/pt_utils.py
# Description: Вспомогательные функции для сохранения и восстановления моделей и объектов.
import glob
import os
import pickle

import torch


def _remove_files(files):
    for f in files:
        return os.remove(f)


def assert_dir_exits(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model, epoch, out_path):
    """
    Сохранение модели в указанную директорию.
    :param model: Модель для сохранения.
    :param epoch: Эпоха, на которой сохраняется модель.
    :param out_path: Директория для сохранения.
    """
    assert_dir_exits(out_path)
    chk_files = glob.glob(out_path + '*.pth')
    _remove_files(chk_files)
    torch.save(model.state_dict(), os.path.join(out_path, str(epoch) + '.pth'))
    print('model saved for epoch: {}'.format(epoch))


def save_objects(obj, epoch, out_path):
    """
    Сохранение объектов (например, состояния обучения) в указанную директорию.
    :param obj: Объекты для сохранения (например, кортеж с метриками).
    :param epoch: Эпоха, на которой сохраняются объекты.
    :param out_path: Директория для сохранения.
    """
    assert_dir_exits(out_path)
    dat_files = glob.glob(out_path + '*.dat')
    _remove_files(dat_files)
    with open(os.path.join(out_path, str(epoch) + '.dat'), 'wb') as output:
        pickle.dump(obj, output)

    print('objects saved for epoch: {}'.format(epoch))


def restore_model(model, out_path, device):
    """
    Восстановление модели из указанной директории.
    :param model: Модель для восстановления.
    :param out_path: Директория, из которой восстанавливается модель.
    :param device: Устройство для загрузки модели (CPU или GPU).
    :return: Восстановленная модель.
    """
    chk_file = glob.glob(out_path + '*.pth')

    if chk_file:
        chk_file = str(chk_file[0])
        print('found model {}, restoring'.format(chk_file))
        # Явно указываем устройство, куда загружаем модель
        model.load_state_dict(torch.load(chk_file, map_location=device))
    else:
        print('Model not found, using untrained model')
    return model


def restore_objects(out_path, default, device):
    """
    Восстановление объектов, сохраненных с использованием GPU, на CPU.
    :param out_path: Директория для восстановления.
    :param default: Значения по умолчанию.
    :param device: Устройство для загрузки объектов (CPU или GPU).
    :return: Восстановленные объекты или значения по умолчанию.
    """
    data_file = glob.glob(out_path + '*.dat')

    if data_file:
        data_file = str(data_file[0])
        print(f'Найден файл данных {data_file}, восстанавливаем...')
        with open(data_file, 'rb') as input_:
            try:
                obj = pickle.load(input_)

                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, torch.Tensor):
                            obj[key] = value.to(device)
                return obj
            except Exception as e:
                print(f"Ошибка при восстановлении объектов: {e}")
                return default
    else:
        return default
