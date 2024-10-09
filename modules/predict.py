import os
import json


import dill
import pandas as pd
import pickle

def predict():
    # Путь к сохраненной модели
    model_path = '../data/models/cars_pipe_202410091734.pkl'

    # Путь к папке с тестовыми данными и папке для сохранения предсказаний
    test_data_path = '../data/test'
    predictions_path = '../data/predictions/predictions.csv'

    # Проверяем, что папка для сохранения предсказаний существует
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    # Загружаем модель
    with open(model_path, 'rb') as file:
        model = dill.load(file)

    # Инициализируем пустой DataFrame для хранения предсказаний
    predictions_df = pd.DataFrame()

    # Читаем каждый файл в тестовой папке и делаем предсказание
    for filename in os.listdir(test_data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(test_data_path, filename)

            with open(file_path, 'r') as f:
                json_data = json.load(f)

            data = pd.json_normalize(json_data)



            pred = model.predict(data).tolist()

            # Сохраняем предсказания в DataFrame
            pred_df = pd.DataFrame([pred], columns=['prediction'])
            pred_df['filename'] = filename
            predictions_df = pd.concat([predictions_df, pred_df], ignore_index=True)

    # Сохраняем предсказания в CSV
    predictions_df.to_csv(predictions_path, index=False)
    print(f'Predictions saved to {predictions_path}')


if __name__ == '__main__':
    predict()
