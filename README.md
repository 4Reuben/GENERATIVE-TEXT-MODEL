# GENERATIVE-TEXT-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: REUBEN THOMAS JOHN

*INTERN ID*:CTO4DN1267

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEEKS

*MENTOR*: NEELA SANTOSH

---

## Project Description

This project demonstrates text generation using an LSTM-based neural network. The objective is to train a deep learning model on a dataset of textual information and enable it to generate human-like text based on an initial prompt.
The model is trained on Wikipedia-style text to understand linguistic patterns and generate sentences that mimic real-world text. The workflow includes tokenization, sequence generation, neural network training, and text prediction.
The project follows a sequential LSTM architecture combined with embedding layers to capture contextual meaning and improve text generation fluency.

The script was developed as part of an internship at **CodTech IT Solutions**, and it serves as Task 4 in the internship’s deliverables.

---

## Technologies & Tools Used

- Python 3.x
- TensorFlow/Keras – For deep learning model development
- NumPy – Array manipulation for efficient processing
- Pickle – To save and load tokenizer models
- Jupyter Notebook / VS Code – Development environments
- Command Line Interface (CLI) – For training and generating text
- Plain Text Files (.txt) – Used as dataset input

---

## File Structure

Codtech IT Solutions/

│

├── Task-2/

│ ├── dataset/

│   ├── wikipedia_small.txt # Training dataset

│ ├── models/

│   ├── text_generation_lstm_model.h5 # Trained LSTM model

│   ├── tokenizer.pkl # Tokenizer object for preprocessed text

│ ├── lstm_text_gen.py # Main Python script

├── README.md # Project documentation (this file)

---

## How It Works

- Input: Reads a text dataset (wikipedia_small.txt) for training.
- Tokenization: Prepares the data using Tokenizer() from Keras.
- Sequence Generation: Converts words into numerical sequences.
- Model Training: Trains an LSTM-based deep learning model on the tokenized text.
- Text Generation: Takes a seed prompt and predicts the next words using the trained model.
- Output: Displays generated text based on the input seed prompt.

---

## What I Did

- Processed textual data using tokenization and padding.
- Developed an LSTM model for sequence prediction.
- Trained the model on a large-scale text dataset.
- Implemented word embedding techniques for better text representation.
- Created a text generation function for dynamic content prediction.
- Saved model weights and tokenizer for reuse without retraining.

---

## What I Didn't Do (Yet)

- No character-level text generation model.
- No integration with a web interface or chatbot.
- No multilingual support—currently works for English text only.

---

## What I Learned

- How LSTM networks capture sequential patterns in text.
- The importance of word embeddings for NLP.
- How to train, save, and reuse deep learning models.
- Debugging model training issues and hyperparameter tuning.
- How AI can be used for creative text generation.

---

## Acknowledgments

Special thanks to **CodTech IT Solutions** for providing this opportunity.
This project enhanced my understanding of deep learning in NLP and how LSTM models can be leveraged for text prediction tasks.

## Output
