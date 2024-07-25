import os
import re
import nltk
import random
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 下载NLTK数据（停用词和punkt分词器）
nltk.download('stopwords')
nltk.download('punkt')

# 文本预处理
def preprocess_text(text):
    # 分词
    words = nltk.word_tokenize(text)
    # 转换为小写
    words = [word.lower() for word in words]
    # 去除标点符号和数字
    words = [word for word in words if re.match(r'^[a-z]+$', word)]
    # 去除停用词
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# 读取数据集
def read_dataset(folder):
    documents = []
    labels = []
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    preprocessed_content = preprocess_text(content)
                    documents.append(preprocessed_content)
                    labels.append(category)
    return documents, labels

# 从训练集中采样构建验证集
def split_train_validation(documents, labels, validation_ratio=0.2):
    return train_test_split(documents, labels, test_size=validation_ratio, random_state=42)

# 构建TF-IDF模型
def build_tfidf_model(documents):
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(documents)
    return X_tfidf, tfidf_vectorizer

# 主程序
if __name__ == "__main__":
    # 读取训练集和测试集
    train_documents, train_labels = read_dataset("20news-bydate-train")
    test_documents, test_labels = read_dataset("20news-bydate-test")

    # 从训练集中采样构建验证集
    train_documents, validation_documents, train_labels, validation_labels = split_train_validation(train_documents, train_labels)

    # 构建TF-IDF模型
    X_train_tfidf, tfidf_vectorizer = build_tfidf_model(train_documents)
    X_validation_tfidf = tfidf_vectorizer.transform(validation_documents)
    X_test_tfidf = tfidf_vectorizer.transform(test_documents)

    # 将文本标签编码为数字
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_validation = label_encoder.transform(validation_labels)
    y_test = label_encoder.transform(test_labels)

    # 构建多层感知机（MLP）模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(set(train_labels)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 在训练集上进行训练
    model.fit(X_train_tfidf.toarray(), y_train, epochs=5, batch_size=32, validation_data=(X_validation_tfidf.toarray(), y_validation))

    # 在验证集上进行测试
    validation_predictions = model.predict_classes(X_validation_tfidf.toarray())
    accuracy_on_validation = accuracy_score(y_validation, validation_predictions)
    macro_f1_on_validation = f1_score(y_validation, validation_predictions, average='macro')
    print(f"Accuracy on validation set: {accuracy_on_validation}")
    print(f"Macro-F1 on validation set: {macro_f1_on_validation}")

    # 在测试集上进行测试
    test_predictions = model.predict_classes(X_test_tfidf.toarray())
    accuracy_on_test = accuracy_score(y_test, test_predictions)
    macro_f1_on_test = f1_score(y_test, test_predictions, average='macro')
    print(f"Accuracy on test set: {accuracy_on_test}")
    print(f"Macro-F1 on test set: {macro_f1_on_test}")

    # 打印分类报告
    print("Classification Report on Test Set:")
    print(classification_report(y_test, test_predictions))
