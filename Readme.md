# Sentiment Analysis using LSTM Neural Networks

This project implements a Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) layers to classify short text reviews as positive or negative. The goal is to explore how LSTMs learn contextual information from sequential text data and build a model capable of automating sentiment detection for customer feedback.

## Table of Contents

- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Tokenization and Padding](#tokenization-and-padding)
  - [Data Splitting](#data-splitting)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)
- [References](#references)

## Project Goal

The primary objectives of this analysis are:

1.  Cleanse and preprocess a sentiment-labeled dataset comprising reviews from IMDB, Amazon, and Yelp.
2.  Convert the text data into a numerical format suitable for a neural network using tokenization and padding.
3.  Build, train, and evaluate an LSTM-based neural network model for binary sentiment classification (positive/negative).
4.  Achieve an accuracy of 80% or higher on an unseen test set, demonstrating the model's effectiveness for real-world review classification.

## Dataset

The dataset used in this project combines reviews from three sources, originally provided by Kotzias et al. (2015):

-   `imdb_labelled.txt`: Movie reviews from IMDB.
-   `amazon_cells_labelled.txt`: Product reviews from Amazon.
-   `yelp_labelled.txt`: Restaurant/business reviews from Yelp.

These datasets were concatenated, resulting in a total of 2748 sentences. Each sentence is labeled with a sentiment score: `1` for positive and `0` for negative. An initial analysis showed the average sentence length is approximately 13 words.

## Methodology

The process involves several steps from raw text to a trained sentiment classification model:

### Data Loading and Preprocessing

-   The three `.txt` files are loaded using Pandas, specifying tab separation and assigning column names "sentence" and "sentiment".
-   The dataframes are concatenated into a single dataframe.
-   Basic analysis of sentence length is performed.

### Tokenization and Padding

-   **Tokenization:** The Keras `Tokenizer` is used to convert the raw text sentences into sequences of integers.
    -   A vocabulary size (`vocab_size`) of 6000 is set.
    -   Words not found in the vocabulary during training are mapped to a special Out-of-Vocabulary (`<OOV>`) token.
    -   The tokenizer automatically converts text to lowercase and removes punctuation.
    -   Example: `"The film was gripping!"` might become `[1, 12, 3, 345]`.
-   **Padding:** To ensure uniform input length for the neural network, sequences are padded or truncated to a `max_length` of 20.
    -   `post` padding adds zeros to the end of shorter sequences.
    -   `post` truncating removes tokens from the end of longer sequences.
    -   The length of 20 was chosen based on data analysis to capture the majority (~95%) of sentence lengths without excessive padding.

### Data Splitting

-   The padded sequences (`X`) and corresponding labels (`y`) are split into training, validation, and test sets using Scikit-learn's `train_test_split`.
-   First, 20% of the data is reserved as the final test set.
-   The remaining 80% is then further split, with 20% of *that portion* used for validation during training.
-   This results in an approximate split of:
    -   64% Training Data
    -   16% Validation Data
    -   20% Test Data

### Model Architecture

A Sequential Keras model is constructed with the following layers:

1.  **Embedding Layer:**
    -   Maps the integer sequences to dense vectors of fixed size (`embedding_dim` = 100).
    -   Learns word representations during training.
    -   Input shape: (`max_length` = 20).
    -   Vocabulary size: (`vocab_size` = 6000).
2.  **LSTM Layer:**
    -   Contains 32 LSTM units.
    -   Processes the sequence of embedding vectors, capturing temporal dependencies and contextual information. Uses `tanh` and `sigmoid` activations internally for gates and cell state updates.
3.  **Dense Layer (Output):**
    -   A single neuron with a `sigmoid` activation function.
    -   Outputs a probability score between 0 and 1, representing the likelihood of the input sentence being positive. A threshold of 0.5 is used for classification (>= 0.5 is positive, < 0.5 is negative).

### Training

-   **Optimizer:** Adam optimizer is used for its adaptive learning rate capabilities.
-   **Loss Function:** `binary_crossentropy` is chosen, suitable for binary classification tasks.
-   **Metrics:** Model performance is monitored using `accuracy`.
-   **Epochs:** Training runs for a maximum of 10 epochs.
-   **Early Stopping:** An `EarlyStopping` callback monitors the `val_loss` (loss on the validation set). Training stops if the validation loss does not improve for 2 consecutive epochs (`patience=2`). The weights from the epoch with the best validation loss are restored (`restore_best_weights=True`) to mitigate overfitting.
-   **Batch Size:** Training is performed using batches of 32 samples.

## Results

-   Training logs indicated signs of overfitting after epoch 4, as validation loss started to increase while training loss continued to decrease.
-   The `EarlyStopping` callback successfully halted training before significant overfitting occurred, restoring the weights from the best epoch based on validation loss.
-   The final evaluation on the unseen **test set yielded an accuracy of 82.55%**, surpassing the project goal of 80%.
-   The training history (loss and accuracy curves for training and validation sets) is plotted to visualize the training dynamics and the effect of early stopping.
-   The trained model is saved to `sentiment_model.h5`.

![Model Loss](loss_plot.png) ![Model Accuracy](accuracy_plot.png) *(Note: Add the actual plot images to your repository and update the paths if needed)*

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    Make sure you have Python 3 installed. It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(You may need to create a `requirements.txt` file based on the imports in the script: `pip freeze > requirements.txt`)*
3.  **Ensure dataset files are present:**
    Place `imdb_labelled.txt`, `amazon_cells_labelled.txt`, and `yelp_labelled.txt` in the same directory as the Python script, or update the file paths in the script accordingly.
4.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your Python file)*
5.  **Outputs:**
    -   Console output will show model summary, training progress, and final test accuracy/loss.
    -   Plots showing training/validation loss and accuracy will be displayed.
    -   The trained model will be saved as `sentiment_model.h5`.

## Dependencies

-   Python 3.x
-   Pandas
-   NumPy
-   TensorFlow / Keras
-   Scikit-learn
-   Matplotlib
-   Seaborn (optional, used for styling plots if integrated)

(See `requirements.txt` for specific versions).

## Ethical Considerations

-   The project utilizes publicly available text data, avoiding the use of private or personally identifiable information.
-   Potential biases inherent in the original datasets (reflecting the language used in reviews) may be learned by the model. Awareness of these potential biases is important when interpreting and deploying the model. The model is designed for general sentiment classification and does not target sensitive attributes.

## Future Work

-   Implement more advanced regularization techniques (e.g., increased dropout, L1/L2 regularization) if overfitting remains a concern.
-   Experiment with different network architectures (e.g., Bidirectional LSTMs, GRUs, Attention mechanisms).
-   Perform hyperparameter tuning (e.g., embedding dimension, LSTM units, learning rate).
-   Explore pre-trained word embeddings (like GloVe or Word2Vec).
-   Further investigate potential biases in the dataset and model predictions.

## References

-   Kotzias, D., Denil, M., De Freitas, N., & Smyth, P. (2015). From Group to Individual Labels using Deep Features. In *KDD ’15: Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 597–606). Association for Computing Machinery. https://doi.org/10.1145/2783258.2783380
-   Chollet, F., & others. (2015). *Keras* [Computer software]. GitHub. https://github.com/keras-team/keras
-   Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. http://www.jmlr.org/papers/v12/pedregosa11a.html
-   TensorFlow Documentation: https://www.tensorflow.org/api_docs/python/tf/keras