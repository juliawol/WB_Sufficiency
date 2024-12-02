# WB QA Sufficiency Model

This repository contains the implementation of a machine learning pipeline for determining whether a product's description contains sufficient information to answer customer questions. The work focuses on building and fine-tuning a classification model while considering various baseline approaches, zero-shot methods, and a final fine-tuned model optimized for precision.


## **Overview**

### **Decision-Making Process**

1. **Problem Statement**:
   - Determine if the provided product description sufficiently answers the customer's question.
   - Output: A binary label (`1` for sufficient, `0` for insufficient) along with a confidence score.

2. **Approaches Considered**:
   - **Baseline**: Traditional TF-IDF vectorization combined with distance measures.
   - **Zero-Shot Classification**: Using pre-trained multilingual models like `xlm-roberta-large-xnli` for zero-shot evaluation.
   - **Fine-Tuned Model**: A custom binary classification model fine-tuned on the `ruBERT` architecture, optimized for precision using Focal Loss and dynamic class weights.

3. **Final Model**:
   - The fine-tuned `ruBERT` model achieved the best precision while maintaining acceptable recall.
   - Focal Loss and dynamic class weights were used to address dataset imbalances and improve performance on minority classes.

### **Project Structure**

- **Notebooks**:
  - `WB_Final_Model.ipynb`: Contains the full pipeline for fine-tuning the final `ruBERT` model, including preprocessing, training, and validation.
  - `WB_QA_sufficiency_baseline.ipynb`: Implements baseline methods like TF-IDF vectorization and distance measures.
  - `WB_Zero_Shot.ipynb`: Explores zero-shot classification using pre-trained models like `xlm-roberta-large-xnli`.

- **Data**:
  - `qa_dataset_labeled.csv`: Labeled dataset used for training and evaluating the models.

- **Files**:
  - `requirements.txt`: Contains all necessary dependencies.

### **Final Model Description**

- **Architecture**:
  - Fine-tuned `ruBERT-base` from `ai-forever`.
  - Binary classification with logits converted to probabilities via softmax.

- **Loss Function**:
  - Focal Loss to penalize easy examples and focus on hard-to-classify cases.

- **Precision Optimization**:
  - Custom thresholds and post-fine-tuning adjustments to prioritize precision.

- **Evaluation Metrics**:
  - Precision, Recall, F1-score, and Accuracy were computed on a held-out validation set and a test set (`qa_card_test.csv`).


### **How to Use**

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Notebooks**:
   - Explore baselines (`WB_QA_sufficiency_baseline.ipynb`).
   - Experiment with zero-shot (`WB_Zero_Shot.ipynb`).
   - Train and validate the final model (`WB_Final_Model.ipynb`).

3. **Evaluate**:
   - You can evaluate the performance by entering ids from the dataset and new questions.
   - You can also check the metrics calculated by using a separate dataset previously unseen by the model.

### **Requirements**
```plaintext
transformers==4.33.2
torch==2.0.1
scikit-learn==1.2.2
pandas==2.0.3
datasets==2.14.4
numpy==1.24.3
```

By the 5th this ML-model will be availiable via demo on HuggingFace. The relevant changes will appear in this file.

#### **Russian Version**

# WB QA Sufficiency Model

Этот репозиторий содержит реализацию модели машинного обучения, натренированную определять, содержит ли описание товара достаточно информации, чтобы ответить на вопросы клиентов. Работа включает построение и обучение модели классификации, анализ базового подхода, методов zero-shot и итоговой дообученной модели, оптимизированной для повышения точности (precision).

---

## **Обзор**

### **Процесс принятия решений**

1. **Постановка задачи**:
   - Определить, содержит ли описание товара достаточно информации, чтобы ответить на вопрос клиента.
   - Результат: бинарная метка (`1` для "достаточно", `0` для "недостаточно") и уровень уверенности.

2. **Подходы**:
   - **Базовый**: Векторизация TF-IDF и метрики расстояний.
   - **Zero-Shot классификация**: Использование предобученных мультиязычных моделей, таких как `xlm-roberta-large-xnli`.
   - **Дообученная модель**: Модель классификации на основе `ruBERT`, дообученная для повышения точности, с использованием Focal Loss и динамических весов классов.

3. **Итоговая модель**:
   - Дообученная модель `ruBERT` показала лучшую точность, сохранив при этом приемлемый уровень полноты.
   - Использовались Focal Loss и динамические веса классов для устранения дисбаланса в данных.


### **Структура проекта**

- **Ноутбуки**:
  - `WB_Final_Model.ipynb`: Полный пайплайн для дообучения модели `ruBERT`, включая предобработку, обучение и валидацию.
  - `WB_QA_sufficiency_baseline.ipynb`: Реализация базовых методов, таких как TF-IDF и метрики расстояний.
  - `WB_Zero_Shot.ipynb`: Исследование методов zero-shot классификации.

- **Данные**:
  - `qa_dataset_labeled.csv`: Размеченный набор данных для обучения и оценки моделей. Изначально сгенерирован автоматически (см. Zero Shot), очищен вручную.

- **Файлы**:
  - `requirements.txt`: Все необходимые зависимости.

### **Описание итоговой модели**

- **Архитектура**:
  - Дообученная модель `ruBERT-base` от `ai-forever`.
  - Двухклассовая классификация с использованием softmax для получения вероятностей.

- **Функция потерь**:
  - Focal Loss для акцентирования внимания на сложных примерах.

- **Оптимизация точности**:
  - Пользовательские пороги и настройки после дообучения для повышения точности.

- **Метрики оценки**:
  - Точность, полнота, F1-score и точность вычислялись на выделенном валидационном наборе и тестовом наборе данных (`qa_card_test.csv`).


### **Как использовать**

1. **Установка**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Запуск ноутбуков**:
   - Базовая модель, точность достигается механически, выставлением порогового значения (`WB_QA_sufficiency_baseline.ipynb`).
   - Генерируем размеченный датасет с помощью zero-shot (`WB_Zero_Shot.ipynb`).
   - Обучаем и валидируем итоговую модель (`WB_Final_Model.ipynb`).

3. **Оценка**:
   - Используйте `qa_card_test.csv` для проверки производительности итоговой модели.
   - `WB_Final_Model.ipynb` предусматривает возможность ввода id (из существующих в базе данных) и любого вопроса.


  К моменту защиты модель будет обернута в оболочку для упрощения тестирования. Изменения будут отражены в этом файле. 
