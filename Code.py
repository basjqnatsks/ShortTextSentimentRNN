import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


df_imdb = pd.read_csv("imdb_labelled.txt", sep="\t", header=None, names=["sentence","sentiment"])
df_amz = pd.read_csv("amazon_cells_labelled.txt", sep="\t", header=None, names=["sentence","sentiment"])
df_yelp = pd.read_csv("yelp_labelled.txt", sep="\t", header=None, names=["sentence","sentiment"])

df = pd.concat([df_imdb, df_amz, df_yelp], axis=0).reset_index(drop=True)


print("Sample data:\n", df.head())
print("\nTotal rows:", len(df))
df['length'] = df['sentence'].apply(lambda x: len(x.split()))
print("\nSentence length stats:\n", df['length'].describe())


sentences = df['sentence'].astype(str).tolist()
labels = df['sentiment'].values

vocab_size = 6000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("Unique words in vocabulary:", len(word_index))

sequences = tokenizer.texts_to_sequences(sentences)

max_length = 20 
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')


X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, 
    labels, 
    test_size=0.20, 
    random_state=42
)


X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.20, 
    random_state=42
)


embedding_dim = 100
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(), 
    metrics=['accuracy']
)

model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    validation_data=(X_val, y_val), 
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

epochs_ran = len(history.history['loss'])
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1, epochs_ran+1), history.history['loss'], label='Train Loss')
plt.plot(range(1, epochs_ran+1), history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1, epochs_ran+1), history.history['accuracy'], label='Train Acc')
plt.plot(range(1, epochs_ran+1), history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
model.save("sentiment_model.h5")
print("\nModel saved to sentiment_model.h5")
