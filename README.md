# Chatbot Project

This repository contains the implementation of a simple AI-powered chatbot built with Python and TensorFlow. The chatbot is designed to recognize user inputs and respond with predefined answers based on a set of training data.

## Features
- **Intent Recognition**: Identifies the user's intent using a trained neural network model.
- **Predefined Responses**: Provides context-aware responses based on intents.
- **Customizable Dataset**: Uses an `intents.json` file to define training patterns and responses.
- **Machine Learning Pipeline**: Implements preprocessing, training, and prediction for the chatbot model.
- **Interactive Chat**: Allows users to interact via a console-based interface with the chatbot.

## Project Structure
- **`chatbot.py`**: The main script to run the chatbot.
- **`training.py`**: Script for training the neural network model using the intent dataset.
- **`intents.json`**: Dataset containing training patterns and responses for different intents.
- **`chatbotmodel.keras`**: The trained neural network model.
- **`words.pkl` & `classes.pkl`**: Pickle files for processed training data.

## How to Use
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model (optional):
   ```bash
   python training.py
   ```
4. Run the chatbot:
   ```bash
   python chatbot.py
   ```

## Customization
- Modify `intents.json` to add or update intents, patterns, and responses.
- Retrain the model using `training.py` after changing the dataset.

## Dependencies
- TensorFlow
- NLTK
- NumPy
- Python 3.7+

## Acknowledgments
- Inspired by conversational AI projects and tutorials.
- Leveraged NLTK for natural language processing and TensorFlow for model training.
