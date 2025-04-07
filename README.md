# Healthcare Chatbot using Deep Learning

This project aims to build a healthcare chatbot that understands and responds to spoken inputs, making it user-friendly and accessible. The chatbot is trained on a dataset of disease symptoms and corresponding diagnoses, enabling it to identify patterns and provide accurate predictions. The use of deep learning techniques ensures that the model continuously improves as it encounters more data, making it a valuable tool in modern healthcare settings.

## Overview
This project demonstrates how to combine Natural Language Processing (NLP), deep learning, and speech recognition to create an interactive and functional healthcare chatbot. The chatbot takes voice input, processes it, predicts the disease, and provides feedback.

## Steps to Build the Chatbot

### 1. Imports and Setup
Key modules used:
- **random**: For generating random numbers.
- **json**: For handling JSON data.
- **pickle**: For serializing and deserializing Python objects.
- **nltk (Natural Language Toolkit)**: For NLP tasks.
- **tensorflow**: For building and training the neural network.
- **WordNetLemmatizer**: For lemmatizing words to ensure consistent analysis.
- **Sequential model (Keras)**: For simplifying the neural network structure.
- **Dense, Dropout, and Activation layers**: For building the network.
- **SGD (Stochastic Gradient Descent)**: For optimizing the model.

### 2. Loading and Preparing Data
- Load JSON data representing diseases, symptoms, and responses.
- Tokenize, lemmatize, and clean the data by removing special characters.
- Split data into words, classes (diseases), and documents (symptom-tag pairs).

### 3. Data Serialization
- Save the processed words and classes using the pickle module for later use.

### 4. Data Conversion for Neural Network
- Convert data into numerical values using a Bag of Words model.
- Shuffle and split data into features (train_x) and labels (train_y).

### 5. Model Training
- Create a Sequential model with several layers.
- Compile the model using the SGD optimizer and train for 200 epochs.
- Save the trained model as `chatbot_model.h5`.

### 6. Making the Chatbot Speak
- Use a pre-trained model to predict diseases based on symptoms.
- Allow user interaction with the bot through voice, and provide responses in both text and voice formats.

## Complete Code
The full code integrates speech recognition, NLP, and the trained model to build a working healthcare chatbot. The chatbot takes voice input, processes it, predicts the disease, and provides feedback.

## Getting Started
To get started with building your own healthcare chatbot, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Prepare your dataset.
4. Follow the steps outlined above to train and deploy your chatbot.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
