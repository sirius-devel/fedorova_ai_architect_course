import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Функция для создания DataFrame и записи в CSV файл
def createFrame(filename='data_theme_39.csv'):
    data = {
        'information technology': [85, 90, 95, 55, 60, 67, 70, 73, 76, 30, 99, 90, 85, 57, 63, 62, 77, 76, 71, 10]*100,
        'management': [70, 80, 82, 80, 85, 75, 76, 71, 78, 60, 80, 75, 61, 70, 65, 63, 60, 72, 59, 30]*100,
        'english': [80, 72, 81, 76, 70, 80, 69, 70, 66, 50, 75, 90, 83, 50, 58, 63, 67, 69, 71, 50]*100,
        'database management systems': [92, 83, 90, 61, 64, 56, 69, 77, 73, 25, 97, 94, 90, 61, 65, 64, 78, 79, 76, 15]*100,
        'algorithms and data structures': [88, 94, 85, 65, 60, 64, 70, 73, 75, 35, 90, 96, 99, 66, 61, 59, 74, 77, 73, 20]*100,
        'cultural studies': [74, 79, 76, 61, 72, 73, 65, 64, 70, 40, 70, 90, 83, 71, 59, 66, 68, 71, 61, 40]*100,
        'final time': [1.75, 2.0, 1.75, 1.25, 1.25, 1.0, 2.0, 1.45, 2.0, 2.0, 1.75, 2.0, 1.75, 1.25, 1.75, 2.0, 2.0, 1.75, 1.5, 0.75]*100,
        'final lab': [91, 95, 93, 61, 63, 60, 73, 71, 76, 31, 99, 91, 92, 62, 65, 66, 77, 75, 72, 10]*100
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


# Функция для загрузки модели
def loadModel(model_filename):
    model = CatBoostClassifier()
    model.load_model(model_filename)
    return model
    
# Функция для тестирования модели и оценки точности
def testClassifying(X_test, y_test, model_filename):
    # Загрузить обученную модель
    model = loadModel(model_filename)
    # Предсказать оценку по итоговой лабораторной на тестовом наборе
    y_pred = model.predict(X_test)
    # Оценить точность модели
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    
# Функция для выбора 3 лучших признаков, обучения и тестирования модели.
# Используется классификация, как в лекциях по этой теме, хотя в рассматриваемом датасете 
# есть определенная неполнота - только 20 различных целых оценок в интервале от 0 до 99, а не 99.
def classify(filename = 'data_theme_39.csv'):
    # Чтение данных из файла
    df = pd.read_csv(filename)
    # Определение признаков и целевой переменной
    X = df.drop(columns=['final lab'])
    y = df['final lab']
    # Выбираем лучшие 3 признака с использованием критерия хи-квадрат
    selector = SelectKBest(chi2, k=3)
    X_new = selector.fit_transform(X, y)
    # Создаем DataFrame из X_new с исходными названиями столбцов
    X_new = pd.DataFrame(X_new, columns=X.columns[selector.get_support()])
    print('перечень трёх лучших признаков для предсказания оценки по итоговой лабораторной:\n', X.columns[selector.get_support()])
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    # Обучение модели CatBoostClassifier
    model = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    # Сохранение модели в файл
    model.save_model('model_theme_39.cbm')
    # Тестируем модель на тестовых данных и оцениваем точность
    test_accuracy = testClassifying(X_test, y_test, 'model_theme_39.cbm')
    print('Точность на тестовых данных =', test_accuracy)
    return selector.get_support()

# Функция для предсказания оценки по итоговой лабораторной для нового студента
def classifyStudent(df, model_filename='model_theme_39.cbm'):
    # Чтение обученной модели из файла
    model = loadModel(model_filename)
    # Использование модели для предсказания
    predicted = model.predict(df)[0]
    return predicted

# Класс для объекта Student
class Student:
    def __init__(self, s1, s2, s3, s4, s5, s6, t):
        self.subject1 = s1
        self.subject2 = s2
        self.subject3 = s3
        self.subject4 = s4
        self.subject5 = s5
        self.subject6 = s6
        self.time = t
        
    def dataFrame(self):
        data = {
        'information technology': [self.subject1],
        'management': [self.subject2],
        'english': [self.subject3],
        'database management systems': [self.subject4],
        'algorithms and data structures': [self.subject5],
        'cultural studies': [self.subject6],
        'final time': [self.time]
        }
        df = pd.DataFrame(data)
        return df



# Пример использования функций
if __name__ == '__main__':
    # Создание DataFrame и запись в CSV файл
    createFrame()
    indices = classify()
    # Пример классификации новой квартиры
    new_student = Student(s1 = 90, s2 = 60, s3 = 40, s4 = 96, s5 = 89, s6 = 50, t = 2.0)
    df = pd.DataFrame(data = np.array([new_student.dataFrame().values[0][indices]]), columns = new_student.dataFrame().columns[indices])
    fp = classifyStudent(df)
    print(f'Прогнозируемая оценка: {fp}')

