import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

DATA_PATH = 'dataset/IMDB_Dataset.csv'
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
MAX_WORDS = 30000
NUM_SAMPLES = 50000
MAX_LEN = 250 # Max sequence length for LSTM
EMBEDDING_DIM = 100  # GloVe 100d
GLOVE_PATH = "glove/glove.6B.100d.txt"

def preprocessing_data(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Contraction Expansion
    text = re.sub(r"n't", " not ", text)
    text = text.replace(" i'm ", " i am ")

    # Negation Handling
    negation_words = ['not', 'no'] # Simpler list after fixing contractions
    punctuation = ['.', ',', ';', '!', '?']
    
    # Case Folding + Negation Tagging
    words = text.lower().split()
    new_words = []
    is_negated = False
    
    for word in words:
        if word in negation_words:
            is_negated = True
            new_words.append(word)
        
        # Reset negation flag at punctuation (end of clause/sentence)
        elif word in punctuation:
            is_negated = False
            new_words.append(word)
        
        # Apply the negation tag to subsequent words
        elif is_negated:
            new_words.append(word + '_NEG')
        
        else:
            new_words.append(word)
            
    return ' '.join(new_words)

def load_data(path):
    df = pd.read_csv(path, nrows=NUM_SAMPLES, skiprows=[1])
    # Handle negations
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(preprocessing_data)
    # Map labels to binary values
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map({'positive': 1, 'negative': 0})
    return df

def prepare_data(df):
    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------------------------
# Classic ML Pipeline (TF-IDF + SVM)
# ------------------------------------------------------------------------------
def classic_ml_pipeline(X_train, X_test, y_train, y_test):
    print("Classic ML Pipeline")
    print("\n" + "="*50)
    print("Start: Classic ML - TF-IDF + SVM")
    print("="*50 + "\n")
    start_time = time.time()

    # Step 1: Feature Engineering with TF-IDF
    vectorizer = TfidfVectorizer(max_features=MAX_WORDS, ngram_range=(1,2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Shape of TF-IDF Train data: {X_train_tfidf.shape[1]}")


    # # Step 2: Model Training with SVM
    # Define parameter grid
    param_grid = {'C': [0.01, 0.1, 1.0, 10, 100]}
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        LinearSVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    # Get best model
    model_svm = grid_search.best_estimator_
    best_c = grid_search.best_params_['C']
    
    print(f"Best C parameter: {best_c}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Step 3: Model Evaluation
    y_pred = model_svm.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    end_time = time.time()

    print("\n--- SVM Results ---")
    print(f"Time Training + Evaluation: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model_svm, vectorizer, accuracy, f1

# ------------------------------------------------------------------------------
# Deep Learning Pipeline (LSTM with GloVe)
# ------------------------------------------------------------------------------
def load_glove_embeddings(glove_path, word_index, max_words, embed_dim):
    embeddings_index = {}

    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.rstrip().split()
            
            if len(values) != embed_dim + 1:
                continue
            
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                continue

    embedding_matrix = np.zeros((max_words, embed_dim))
    for word, i in word_index.items():
        if i < max_words and word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]

    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embedding_matrix

def deep_learning_pipepline(X_train, X_test, y_train, y_test):
    print("Deep Learning Pipeline")
    print("\n" + "="*50)
    print("Start: Deep Learning - CNN + BiLSTM (GloVe)")
    print("="*50 + "\n")
    start_time = time.time()
    
    # Step 1: Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Padding sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # Step 2: Building the Improved LSTM Model
    model_lstm = Sequential()

    # Pretrained GloVe embedding (frozen for stability)
    embedding_matrix = load_glove_embeddings(
        GLOVE_PATH,
        tokenizer.word_index,
        MAX_WORDS,
        EMBEDDING_DIM
    )

    # Freeze embeddings to prevent overfitting on IMDB
    model_lstm.add(Embedding(input_dim=MAX_WORDS, 
                             output_dim=EMBEDDING_DIM, 
                             input_length=MAX_LEN,
                             weights=[embedding_matrix],
                             trainable=False))
    model_lstm.add(Dropout(0.2))

    # CNN Feature Extractor
    model_lstm.add(Conv1D(64, kernel_size=5, activation='relu'))
    model_lstm.add(MaxPooling1D(pool_size=2))

    # BiLSTM Layer 1
    model_lstm.add(Bidirectional(
        LSTM(64, dropout=0.3, return_sequences=True)
    ))

    # BiLSTM Layer 2
    model_lstm.add(Bidirectional(
        LSTM(32, dropout=0.3)
    ))

    # Dense Layers for deeper learning
    model_lstm.add(Dense(64, activation='relu'))
    model_lstm.add(Dropout(0.4))

    # Output layer
    model_lstm.add(Dense(1, activation='sigmoid'))

    # Gradient clipping to stabilize LSTM training
    model_lstm.compile(
        optimizer=Adam(
            learning_rate=1e-3,
            clipnorm=1.0
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model_lstm.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )

    # Step 3: Train/Validation split + Model Training
    X_train_pad, X_val_pad, y_train, y_val = train_test_split(
        X_train_pad, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    
    history = model_lstm.fit(
        X_train_pad, y_train,
        epochs=25,
        batch_size=64,
        validation_data=(X_val_pad, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Step 4: Evaluation
    _, accuracy = model_lstm.evaluate(X_test_pad, y_test, verbose=0)
    y_pred_prob = model_lstm.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, average='weighted')

    end_time = time.time()

    print("\n--- Optimized CNN-BiLSTM Results ---")
    print(f"Time Training + Evaluation: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Final epoch reached: {len(history.history['accuracy'])}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Step 5: Enhanced Graphical Visualization (Loss/Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Optimized CNN-BiLSTM: Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Optimized CNN-BiLSTM: Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

    input("Press Enter to continue...")

    return model_lstm, tokenizer, accuracy, f1

if __name__ == "__main__":
    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(data)

    model_svm, vectorizer, acc_svm, f1_svm = classic_ml_pipeline(X_train, X_test, y_train, y_test)
    model_lstm, tokenizer, acc_lstm, f1_lstm = deep_learning_pipepline(X_train, X_test, y_train, y_test)

    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)

    results = pd.DataFrame({
        'Model': ['TF-IDF + SVM (Classic ML)', 'LSTM (Deep Learning)'],
        'Accuracy': [acc_svm, acc_lstm],
        'F1 Score': [f1_svm, f1_lstm]
    }).set_index('Model')

    print(results.to_markdown())

    if acc_svm > acc_lstm:
        print("\nThe Classic ML model (TF-IDF + SVM) performed better.")
    else:
        print("\nThe Deep Learning model (LSTM) performed better.")