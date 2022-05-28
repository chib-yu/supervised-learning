<h1>Содержание<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Открытие-файла-и-изучение-данных" data-toc-modified-id="Открытие-файла-и-изучение-данных-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Открытие файла и изучение данных</a></span></li><li><span><a href="#Подготовка-признаков" data-toc-modified-id="Подготовка-признаков-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Подготовка признаков</a></span></li><li><span><a href="#Создание-обучающей,-валидационной-и-тестовой-выборок" data-toc-modified-id="Создание-обучающей,-валидационной-и-тестовой-выборок-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Создание обучающей, валидационной и тестовой выборок</a></span></li><li><span><a href="#Выводы-по-п.-1" data-toc-modified-id="Выводы-по-п.-1-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Выводы по п. 1</a></span></li></ul></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span><ul class="toc-item"><li><span><a href="#Модель-решающего-дерева" data-toc-modified-id="Модель-решающего-дерева-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Модель решающего дерева</a></span></li><li><span><a href="#Модель-случайного-леса" data-toc-modified-id="Модель-случайного-леса-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Модель случайного леса</a></span></li><li><span><a href="#Логистическая-регрессия" data-toc-modified-id="Логистическая-регрессия-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Логистическая регрессия</a></span></li><li><span><a href="#Сравнение-моделей-разного-типа" data-toc-modified-id="Сравнение-моделей-разного-типа-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Сравнение моделей разного типа</a></span></li><li><span><a href="#Выводы-по-п.-2" data-toc-modified-id="Выводы-по-п.-2-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Выводы по п. 2</a></span></li></ul></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span><ul class="toc-item"><li><span><a href="#Выводы-по-п.-3" data-toc-modified-id="Выводы-по-п.-3-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Выводы по п. 3</a></span></li></ul></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span><ul class="toc-item"><li><span><a href="#Выводы-по-п.-4" data-toc-modified-id="Выводы-по-п.-4-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Выводы по п. 4</a></span></li></ul></li><li><span><a href="#Выводы-по-проекту" data-toc-modified-id="Выводы-по-проекту-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Выводы по проекту</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# Исследование: Отток клиентов

**Задача:** Спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. 

**Данные:** Исторические данные о поведении клиентов и расторжении договоров с банком. Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

**Цель:** Построить модель с предельно большим значением *F1*-меры. Довести метрику до 0.59. Проверить *F1*-меру на тестовой выборке, дополнительно измерить *AUC-ROC*, сравнить её значение с *F1*-мерой.

## Подготовка данных

### Открытие файла и изучение данных


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
```


```python
try:
    df = pd.read_csv('/datasets/Churn.csv') # server path
except:
    df = pd.read_csv('C:/Users/datasets/Churn.csv') # local path

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 14 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   RowNumber        10000 non-null  int64  
     1   CustomerId       10000 non-null  int64  
     2   Surname          10000 non-null  object 
     3   CreditScore      10000 non-null  int64  
     4   Geography        10000 non-null  object 
     5   Gender           10000 non-null  object 
     6   Age              10000 non-null  int64  
     7   Tenure           9091 non-null   float64
     8   Balance          10000 non-null  float64
     9   NumOfProducts    10000 non-null  int64  
     10  HasCrCard        10000 non-null  int64  
     11  IsActiveMember   10000 non-null  int64  
     12  EstimatedSalary  10000 non-null  float64
     13  Exited           10000 non-null  int64  
    dtypes: float64(3), int64(8), object(3)
    memory usage: 1.1+ MB


**Описание данных**

Признаки:
- RowNumber — индекс строки в данных
- CustomerId — уникальный идентификатор клиента
- Surname — фамилия
- CreditScore — кредитный рейтинг
- Geography — страна проживания
- Gender — пол
- Age — возраст
- Tenure — сколько лет человек является клиентом банка
- Balance — баланс на счёте
- NumOfProducts — количество продуктов банка, используемых клиентом
- HasCrCard — наличие кредитной карты
- IsActiveMember — активность клиента
- EstimatedSalary — предполагаемая зарплата

Целевой признак:
- Exited — факт ухода клиента


```python
df.head(n=5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1.0</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8.0</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2.0</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Датафрейм содержит 10000 записей. 

Есть пропуски в столбце 'Tenure'. Кроме того, данные в нем, вероятно, следует преобразовать в целочисленный тип.

Данные в столбцах 'RowNumber', 'CustomerId' и 'Surname' не связаны с целевым признаком и, вероятно, будут мешать проведению обучения.

Имеются столбцы с категориальными признаками:
- Geography — страна проживания;
- Gender — пол.

### Подготовка признаков


```python
df['Tenure'].isna().sum()
```




    909




```python
df['Tenure'].unique()
```




    array([ 2.,  1.,  8.,  7.,  4.,  6.,  3., 10.,  5.,  9.,  0., nan])




```python
df['Tenure'].hist(bins=11)
```




    <AxesSubplot:>




    
![png](output_13_1.png)
    


В столбце 'Tenure' 909 пропущенных значений, при этом необычно мало значений, равных 0. Скорее всего, пропущенные значения соответствуют нулям.

Все не пропущенные значения являются целочисленными.


```python
df['Tenure'] = df['Tenure'].fillna(0).astype('int')
```

Проведено заполнение нулями пропущенных значений в столбце 'Tenure' и замена его типа на целочисленный.


```python
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```

Из данных удалены неинформативные признаки, которые могут помешать проведению обучения:
- RowNumber — индекс строки в данных;
- CustomerId — уникальный идентификатор клиента;
- Surname — фамилия.


```python
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
      <th>Geography_Germany</th>
      <th>Geography_Spain</th>
      <th>Gender_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Выполнено One-Hot Encoding категориальных признаков:
- Geography — страна проживания;
- Gender — пол.

### Создание обучающей, валидационной и тестовой выборок


```python
features = df.drop(['Exited'], axis=1)
target = df['Exited']

features_train_valid, features_test, target_train_valid, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345)

features_train, features_valid, target_train, target_valid = train_test_split(
    features_train_valid, target_train_valid, test_size=0.25, random_state=12345)

print("features_train.shape:", features_train.shape)
print("features_valid.shape:", features_valid.shape)
print("features_test.shape:", features_test.shape)
```

    features_train.shape: (6000, 11)
    features_valid.shape: (2000, 11)
    features_test.shape: (2000, 11)


Данные разделены на целевые признаки (факт ухода клиента) и признаки (все остальное). 

Выборка случайным образом разделена на
- обучающую (60% - 6000 записей);
- валидационную (20% - 2000 записей);
- тестовую (20% - 2000 записей).

### Выводы по п. 1

Загружен файл с данными о 10 000 клиентах банка. 

Пропуски в столбце Tenure заполнены нулями, его тип заменен на целочисленный.

Из данных удалены неинформативные признаки, которые могут помешать проведению обучения: RowNumber, CustomerId, Surname.

Выполнено One-Hot Encoding категориальных признаков Geography, Gender.

Данные разделены на целевые признаки (факт ухода клиента) и признаки (все остальное). 

Выборка случайным образом разделена на обучающую (60%), валидационную (20%), тестовую (20%).

## Исследование задачи


```python
best_models = []
best_results = []
best_roc_aucs = []
```

Подготовлены списки для сохранения лучших моделей каждого типа.

### Модель решающего дерева


```python
best_model = None
best_result = 0
best_roc_auc = 0
best_depth = None


for depth in range(1, 31):
    
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth) # обучение модели с заданной глубиной дерева
    model.fit(features_train, target_train) # обучение модели
    predicted_valid = model.predict(features_valid) # получение предсказания модели
    result = f1_score(target_valid, predicted_valid) # подсчет качества модели на валидационной выборке
    roc_auc = roc_auc_score(target_valid, predicted_valid)
    if result > best_result:
        best_model = model # сохранение наилучшей модели
        best_result = result # сохранение наилучшего значения F1-меры на валидационных данных
        best_roc_auc = roc_auc # сохранение значения ROC AUC наилучшей модели
        best_depth = depth # сохранение параметра max_depth наилучшей модели
        
best_models.append(best_model)
best_results.append(best_result)
best_roc_aucs.append(best_roc_auc)

print("F1-мера лучшей модели решающего дерева на валидационной выборке:", best_result)
print("ROC AUC мера лучшей модели решающего дерева на валидационной выборке:", best_roc_auc)
print("Глубина лучшей модели решающего дерева:", best_depth)
```

    F1-мера лучшей модели решающего дерева на валидационной выборке: 0.5488958990536277
    ROC AUC мера лучшей модели решающего дерева на валидационной выборке: 0.7010645044896117
    Глубина лучшей модели решающего дерева: 7


### Модель случайного леса


```python
best_model = None
best_result = 0
best_roc_auc = 0
best_depth = None
best_est = None

for est in range(10, 101, 10):
    for depth in range(1, 31):
        model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth) # обучение модели с заданным количеством деревьев
        model.fit(features_train, target_train) # обучение модели на тренировочной выборке
        predicted_valid = model.predict(features_valid) # получение предсказания модели
        result = f1_score(target_valid, predicted_valid) # подсчет качества модели на валидационной выборке
        if result > best_result:
            best_model = model # сохранение наилучшей модели
            best_result = result # сохранение наилучшего значения F1-меры на валидационных данных
            best_roc_auc = roc_auc # сохранение значения ROC AUC наилучшей модели
            best_est = est # сохранение параметра n_estimators наилучшей модели
            best_depth = depth # сохранение параметра max_depth наилучшей модели
        
best_models.append(best_model)
best_results.append(best_result)
best_roc_aucs.append(best_roc_auc)

print("F1-мера наилучшей модели случайного леса на валидационной выборке:", best_result)
print("ROC AUC мера лучшей модели случайного леса на валидационной выборке:", best_roc_auc)
print("Число деревьев лучшей модели случайного леса:", best_est)
print("Глубина деревьев лучшей модели случайного леса:", best_depth)
```

    F1-мера наилучшей модели случайного леса на валидационной выборке: 0.5736925515055468
    ROC AUC мера лучшей модели случайного леса на валидационной выборке: 0.6794414093359126
    Число деревьев лучшей модели случайного леса: 40
    Глубина деревьев лучшей модели случайного леса: 23


### Логистическая регрессия


```python
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train) # обучение модели на тренировочной выборке
predicted_valid = model.predict(features_valid) # получение предсказания модели
result = f1_score(target_valid, predicted_valid) # F1-мера качества модели на валидационной выборке
roc_auc = roc_auc_score(target_valid, predicted_valid) # ROC AUC мера качества модели на валидационной выборке

best_models.append(model)
best_results.append(result)
best_roc_aucs.append(roc_auc)

print("F1-мера модели логистической регрессии на валидационной выборке:", result)
print("ROC AUC мера модели логистической регрессии на валидационной выборке:", roc_auc)
```

    F1-мера модели логистической регрессии на валидационной выборке: 0.10290827740492169
    ROC AUC мера модели логистической регрессии на валидационной выборке: 0.519156948049574


### Сравнение моделей разного типа


```python
very_best_result = max(best_results)
very_best_model = best_models[best_results.index(very_best_result)]
very_best_roc_auc = best_roc_aucs[best_results.index(very_best_result)]
print('Наилучшая из опробованных моделей:', very_best_model, 
      '\nее F1-мера -', very_best_result, 
      '\nее ROC AUC мера -', very_best_roc_auc)
```

    Наилучшая из опробованных моделей: RandomForestClassifier(max_depth=23, n_estimators=40, random_state=12345) 
    ее F1-мера - 0.5736925515055468 
    ее ROC AUC мера - 0.6794414093359126


### Выводы по п. 2

Были обучены 3 типа моделей с различными гиперпараметрами
- решающее дерево глубиной от 1 до 30;
- модель случайного леса с числом оценщиков от 10 до 100 с шагом 10 и максимальной глубиной 30;
- модель логистической регрессии.

Наилучшее значение F1-меры на валидационной выборке дала модель случайного леса с 40 деревьями и максимальной глубиной 23. Ее F1-score = 0.5737, ROC AUC мера - 0.6794. Целевое значение F1-меры 0.59 не достигнуто.

## Борьба с дисбалансом


```python
df['Exited'].value_counts()
```




    0    7963
    1    2037
    Name: Exited, dtype: int64



Имеется дисбаланс в данных - в целевом столбце единица встречается в 4 раза реже, чем 0. Решено провести upsampling данных с единицей в целевом столбце.


```python
model = LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print("F1:", f1_score(target_valid, predicted_valid))
print("ROC AUC:", roc_auc_score(target_valid, predicted_valid))
```

    F1: 0.4754521963824289
    ROC AUC: 0.6994296786458525


У модели логистической регрессии с параметром class_weight='balanced' существенно повысилось значение F1-меры, однако оно все равно не достигает целевого значения 0.59.


```python
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train, target_train, 4)
```

### Выводы по п. 3
- Имеется дисбаланс в данных - в целевом столбце единица встречается в 4 раза реже, чем 0. 
- У модели логистической регрессии с параметром class_weight='balanced' существенно повысилось значение F1-меры, однако оно все равно не достигает целевого значения 0.59.
- Проведен upsampling данных с единицей в целевом столбце. Решено исследовать модель случайного леса, лучше всего показавшую себя на исходных данных.

## Тестирование модели


```python
very_best_model.fit(features_upsampled, target_upsampled) # обучение модели на тренировочной выборке
predicted_valid = very_best_model.predict(features_valid) # получение предсказания модели
result = f1_score(target_valid, predicted_valid) # подсчет качества модели на валидационной выборке 
roc_auc = roc_auc_score(target_valid, predicted_valid) # подсчет ROC AUC меры модели на валидационной выборке 

print("F1-мера наилучшей модели случайного леса после переобучения на сбалансированной выборке:", result)
print("ROC AUC мера наилучшей модели случайного леса после переобучения на сбалансированной выборке:", roc_auc)
```

    F1-мера наилучшей модели случайного леса после переобучения на сбалансированной выборке: 0.5859154929577465
    ROC AUC мера наилучшей модели случайного леса после переобучения на сбалансированной выборке: 0.7314911805238754


F1-мера модели случайного леса с гиперпараметрами, давшими наилучший результат на исходной тестовой выборке, после обучения на сбалансированной выборке увеличилась с 0.5737 до 0.5859, но не достигла целевого значения 0.59.

Решено оставить модель случайного леса, но заново провести подбор гиперпараметров.


```python
best_model = None
best_result = 0
best_roc_auc = 0
best_depth = None
best_est = None

for est in range(10, 101, 10):
    for depth in range(1, 31):
        model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth) # обучение модели с заданным количеством деревьев
        model.fit(features_upsampled, target_upsampled) # обучение модели на тренировочной выборке
        predicted_valid = model.predict(features_valid) # получение предсказания модели
        result = f1_score(target_valid, predicted_valid) # подсчет качества модели на валидационной выборке
        if result > best_result:
            best_model = model # сохранение наилучшей модели
            best_result = result # сохранение наилучшего значения F1-меры на валидационных данных
            best_roc_auc = roc_auc # сохранение значения ROC AUC наилучшей модели
            best_est = est # сохранение параметра n_estimators наилучшей модели
            best_depth = depth # сохранение параметра max_depth наилучшей модели
        
print("F1-мера наилучшей модели случайного леса на валидационной выборке:", best_result)
print("ROC AUC мера лучшей модели случайного леса на валидационной выборке:", best_roc_auc)
print("Число деревьев лучшей модели случайного леса:", best_est)
print("Глубина деревьев лучшей модели случайного леса:", best_depth)
```

    F1-мера наилучшей модели случайного леса на валидационной выборке: 0.6040268456375838
    ROC AUC мера лучшей модели случайного леса на валидационной выборке: 0.7314911805238754
    Число деревьев лучшей модели случайного леса: 60
    Глубина деревьев лучшей модели случайного леса: 15


Наилучшее значение F1-меры на валидационной выборке (0.6040) получено с помощью модели случайного леса с числом деревьев 60 и максимальной глубиной 15.


```python
predicted_test = model.predict(features_test) # получение предсказания модели
result = f1_score(target_test, predicted_test) # подсчет качества модели на тестовой выборке
roc_auc = roc_auc_score(target_test, predicted_test) # подсчет ROC AUC меры на тестовой выборке
print("F1-мера лучшей модели случайного леса на тестовой выборке:", result)
print("ROC AUC мера лучшей модели случайного леса на тестовой выборке:", roc_auc)
```

    F1-мера лучшей модели случайного леса на тестовой выборке: 0.5947368421052632
    ROC AUC мера лучшей модели случайного леса на тестовой выборке: 0.7306255592395683


Значение F1-меры полученной модели на тестовой выборке равно 0.5947 и превосходит целевое значение 0.59.
Значение ROC AUC меры полученной модели на тестовой выборке равно 0.7306.


```python
features_train_valid_upsampled, target_train_valid_upsampled = upsample(features_train_valid, target_train_valid, 4)

new_model = RandomForestClassifier(random_state=12345, n_estimators=60, max_depth=15) # обучение модели с заданным количеством деревьев
new_model.fit(features_train_valid_upsampled, target_train_valid_upsampled) # обучение модели на тренировочной выборке
new_predicted_valid = model.predict(features_valid) # получение предсказания модели
result = f1_score(target_valid, predicted_valid) # подсчет качества модели на валидационной выборке

print("F1-мера модели случайного леса на тестовой выборке:", result)
print("ROC AUC мера модели случайного леса на тестовой выборке:", roc_auc)
```

    F1-мера модели случайного леса на тестовой выборке: 0.5770308123249299
    ROC AUC мера модели случайного леса на тестовой выборке: 0.7306255592395683


Модель с наилучшими гиперпараметрами была обучена на объединении тестовых и валидационных данных, подвергнутых upsampling. F1-мера данной модели (0.5770) оказалась ниже, чем у модели, обученной на тестовой выборке. Возможно, это произошло из-за влияния случайных факторов в ходе создания выборки.

### Выводы по п. 4
- F1-мера модели случайного леса с гиперпараметрами, давшими наилучший результат на исходной тестовой выборке, после обучения на сбалансированной выборке увеличилась, но не достигла целевого значения 0.59. Решено оставить модель случайного леса, но заново провести подбор гиперпараметров.
- С помощью валидационной выборки опрелены наилучшие гиперпараметры модели случайного леса: число деревьев 60, максимальная глубина 15.
- Значение F1-меры данной модели на тестовой выборке равно 0.5947 и превосходит целевое значение 0.59.
Значение ROC AUC меры полученной модели на тестовой выборке равно 0.7306.
- Модель с наилучшими гиперпараметрами была обучена на сбалансированном объединении обучающих и валидационных данных. F1-мера данной модели (0.5770) оказалась ниже, чем у модели, обученной на тестовой выборке. Возможно, это произошло из-за влияния случайных факторов в ходе создания выборки.

## Выводы по проекту

- Загружен файл с данными о 10 000 клиентах банка.
- В столбце Tenure заполнены нулями пропуски, его тип заменен на целочисленный.
- Удалены неинформативные признаки, которые могут помешать проведению обучения: RowNumber, CustomerId, Surname.
- Выполнено One-Hot Encoding категориальных признаков Geography, Gender.
- Данные разделены на целевые признаки (факт ухода клиента) и признаки (все остальное). 
- Выборка разделена на обучающую (60%), валидационную (20%), тестовую (20%).
- Обучены 3 типа моделей с различными гиперпараметрами: решающее дерево глубиной, модель случайного леса, модель логистической регрессии.
- Наилучшее значение F1-меры на валидационной выборке дала модель случайного леса с 40 деревьями и максимальной глубиной 23. Целевое значение F1-меры 0.59 не достигнуто.
- Обнаружен дисбаланс классов - в целевом столбце единица встречается в 4 раза реже, чем 0. 
- У модели логистической регрессии с параметром class_weight='balanced' существенно повысилось значение F1-меры, однако оно все равно не достигает целевого значения 0.59.
- Проведен upsampling данных с единицей в целевом столбце. Решено исследовать модель случайного леса, лучше всего показавшую себя на исходных данных.
- С помощью сбалансированной валидационной выборки опрелены наилучшие гиперпараметры модели случайного леса: число деревьев 60, максимальная глубина 15.
- Значение F1-меры данной модели на тестовой выборке равно 0.5947 и превосходит целевое значение 0.59.
Значение ROC AUC меры полученной модели на тестовой выборке равно 0.7306.
- Для всех исследованных случаев значение ROC AUC меры оказывалось несколько выше значения F1-меры.

## Чек-лист готовности проекта

Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

- [x]  Jupyter Notebook открыт
- [x]  Весь код выполняется без ошибок
- [x]  Ячейки с кодом расположены в порядке исполнения
- [x]  Выполнен шаг 1: данные подготовлены
- [x]  Выполнен шаг 2: задача исследована
    - [x]  Исследован баланс классов
    - [x]  Изучены модели без учёта дисбаланса
    - [x]  Написаны выводы по результатам исследования
- [x]  Выполнен шаг 3: учтён дисбаланс
    - [x]  Применено несколько способов борьбы с дисбалансом
    - [x]  Написаны выводы по результатам исследования
- [x]  Выполнен шаг 4: проведено тестирование
- [x]  Удалось достичь *F1*-меры не менее 0.59
- [x]  Исследована метрика *AUC-ROC*
