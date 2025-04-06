import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, jsonify

# Загрузка необходимых данных для nltk
nltk.download('punkt')
nltk.download('stopwords')

# Инициализация Flask-приложения
app = Flask(__name__)

# Загружаем датасет
file_path = "vgchartz-2024.csv"
data = pd.read_csv(file_path, encoding='latin1')

# Удаляем пустые значения в ключевых колонках is used to remove rows from the dataset where any of the specified columns contain missing values
data.dropna(subset=['title', 'genre', 'publisher', 'developer', 'total_sales'], inplace=True)

# Инициализация инструментов обработки текста Loads a list of common words (stopwords) in English, such as "the," "is," "and," "in," "to,"
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Функция предобработки текста
def preprocess_text(text):
    text = str(text).lower()  # Приводим к нижнему регистру
    tokens = word_tokenize(text)  # Токенизация
    tokens = [word for word in tokens if word.isalnum()]  # remove punctuation marks
    tokens = [word for word in tokens if word not in stop_words]  # Убираем стоп-слова
    tokens = [stemmer.stem(word) for word in tokens]  # Стемминг приводим к корню
    return ' '.join(tokens)  # Объединяем обратно в текст


# Создаем текстовую колонку с информацией об игре
data['processed_text'] = data.apply(lambda row: f"{row['title']} {row['genre']} {row['publisher']} {row['developer']}",
                                    axis=1)
data['processed_text'] = data['processed_text'].apply(preprocess_text)

# TF-IDF векторизация
#converts text into numerical form
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'])

# Извлечение ключевых слов
num_keywords = 5  # Количество ключевых слов
feature_names = tfidf_vectorizer.get_feature_names_out()


def extract_top_keywords(row):
    document_vector = tfidf_matrix[row].toarray().flatten() #extract vector convert into 1d matrix
    top_indices = document_vector.argsort()[::-1][:num_keywords]  #sorts the TF-IDF scores and returns the indices of words, from lowest to highest.
    return [feature_names[i] for i in top_indices] #Returns the top 5 words with the highest TF-IDF scores for that game.



# Применяем функцию извлечения ключевых слов
data['top_keywords'] = [extract_top_keywords(i) for i in range(len(data))]

# Векторизация ключевых слов для кластеризации
tfidf_vectorizer_keywords = TfidfVectorizer(max_features=1000) #Creates another TF-IDF vectorizer, but this time for top keywords, not full text descriptions.
tfidf_matrix_keywords = tfidf_vectorizer_keywords.fit_transform(data['top_keywords'].apply(' '.join)) #Transforming the keywords into a numerical matrix
#Learns the vocabulary from the top keywords.

# Кластеризация с K-Means
num_clusters = 5 #Defines the number of clusters (groups) the K-Means algorithm will create.
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # num clusters (5), ensures consistency
data['cluster'] = kmeans.fit_predict(tfidf_matrix_keywords) #Adds a new column called cluster where each game is assigned a cluster ID (from 0 to 4, since there are 5 clusters).


# Функция поиска игр по ключевым словам
def search_games(query):
    if not query.strip():
        return pd.DataFrame()  # Возвращаем пустой DataFrame, если запрос пустой

    query = preprocess_text(query) #preprocessing token i td
    query_vector = tfidf_vectorizer.transform([query]) #converts into tfidf
    similarities = tfidf_matrix.dot(query_vector.T).toarray().flatten() #calculating similarities between query and game data
    top_indices = similarities.argsort()[::-1][:5] #finding top 5 most similair genres
    return data.iloc[top_indices] # return them


# Главная страница с кластеризованными играми
@app.route('/')
def index():
    #Grouping games by their clusters
    clustered_games = { #Grouping games by their clusters
        f'Cluster {i}': data[data['cluster'] == i][['title', 'total_sales']].sort_values(by='total_sales',
                                                                                         ascending=False).head().to_dict(
            orient='records')
        for i in range(num_clusters)
    }
    return render_template('index.html', clustered_articles=clustered_games) #Rendering the HTML template


# search query handler
@app.route('/search', methods=['POST']) #Flask route decorator that specifies the URL path for search functionality.
def search():
    query = request.form['query'] #Extracts the search query submitted by the user from the form data
    top_games = search_games(query) #Searching for the most similar games
#formatting the search results
    search_results = [
        {'Title': row['title'], 'Genre': row['genre'], 'Publisher': row['publisher'], 'Developer': row['developer'],
         'Total Sales': row['total_sales']}
        for _, row in top_games.iterrows()
    ]

    return jsonify(search_results) #retund result as json


# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)
