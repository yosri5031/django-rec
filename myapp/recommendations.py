import pandas as pd
import re
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove HTML entities
    text = re.sub(r'&[^;]*;', '', text)
    # Convert to lowercase
    text = text.lower()
    preprocessed_text = text
    return preprocessed_text

def recommend_products(preprocessed_answers):
    # Load the product data
    df = pd.read_csv('output.csv')

    # Preprocess the product titles
    df['Title'] = df['Title'].apply(preprocess_text)

    # Convert the product titles into sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Title'])
    sequences = tokenizer.texts_to_sequences(df['Title'])

    # Pad the sequences
    max_sequence_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df.index, test_size=0.2, random_state=42)
    # Build and train the model
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_sequence_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(df), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16)

    # Generate recommendations for the user's answers
    preprocessed_answers_sequence = tokenizer.texts_to_sequences([preprocessed_answers])
    preprocessed_answers_padded = pad_sequences(preprocessed_answers_sequence, maxlen=max_sequence_length)
    predictions = model.predict(preprocessed_answers_padded)
    recommended_indices = predictions.argsort()[0][::-1][:5]
    recommended_products = df.iloc[recommended_indices]

    # Calculate similarity values
    similarity_values = cosine_similarity(preprocessed_answers_padded, X_test)

    return recommended_products, similarity_values

def generate_plot(similarity_values, recommended_products):
    # Generate a plot to visualize the similarity values
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=similarity_values[0], y=recommended_products['Title'])
    plt.xlabel('Similarity')
    plt.ylabel('Product Title')
    plt.title('Product Similarity')
    plt.tight_layout()
    plt.savefig('plot.png')