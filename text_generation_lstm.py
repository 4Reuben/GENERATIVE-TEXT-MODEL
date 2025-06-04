# LSTM TEXT GENERATION MODEL

import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define filenames
MODEL_PATH = "text_generation_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# Check if trained model exists
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    print("✅ Loading trained model and tokenizer...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    total_words = len(tokenizer.word_index) + 1
    max_seq_len = model.input_shape[1] + 1  # Adjust sequence length

else:
    print("🚀 Training new model...")

    # STEP 1: Load dataset
    with open("dataset/wikipedia_small.txt", "r", encoding='utf-8') as f:
        data = f.read().lower()

    # STEP 2: Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    total_words = len(tokenizer.word_index) + 1

    # STEP 3: Create input sequences
    input_sequences = []
    for line in data.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram = token_list[:i+1]
            input_sequences.append(n_gram)

    # STEP 4: Pad sequences
    max_seq_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

    # Split into input (X) and label (y)
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = np.array(y)

    # STEP 5: Create the model
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_seq_len - 1))
    model.add(LSTM(256))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # STEP 6: Train the model
    model.fit(X, y, epochs=300, verbose=1)

    # Save the model & tokenizer
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print("\n✅ Model training complete. Saved for future use.")

# STEP 7: Text generation function (Unlimited prompts)
def generate_text(seed_text, next_words=40):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

# STEP 8: Continuous prompt input
while True:
    prompt = input("\nEnter a topic prompt (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        print("\n👋 Exiting text generation...")
        break
    print("\nGenerated Paragraph:\n")
    print(generate_text(prompt, next_words=40))
