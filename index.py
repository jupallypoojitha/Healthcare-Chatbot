
import random
import json
import pickle
 
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
 
import numpy as np
Brief Explanation of some Imported modules – 
nltk – Natural Language Toolkit (NLTK) is a Python module widely used to do tasks related to Natural Language Processing (NLP).
WordNetLemmatizer – It is a built-in function of the wordnet (which is part of NLTK, we have installed it above). We will use this to lemmatize the intents we will use later. Lemmatization is a process that groups together the different inflected forms of a word so they can be analyzed as a single item.
Sequential – Sequential is the model which we will use here to make this chatbot less complex. A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
Dense, Activation, Dropout – Dense is a fully connected layer in Neural Network whereas Dropout is not, dropout ignores a random set of neurons to prevent overfitting. Activation is the Activation function that we will use in our code to decide which neuron will be activated and which will not.
SGD – Stochastic Gradient Descent (SGD) is an Optimizer present in Keras which is an iterative method for optimizing an objective function with suitable smoothness properties.
After Importing all the necessary modules it is time to load up the dataset i.e the JSON file. The format of the JSON file is given below – 

{
    "tag" : "Name of the Disease",
    "patterns" : ["comma separated symptoms"],
    "responses" : ["Answer user will receive i.e the disease user might have"]
  }
In place of the tag, patterns, and responses the user can give any name but make sure the same name is being used further in the code. The above snippet is just a structure of one Disease and its symptoms and responses. The entire structure would be as shown below.

{"intents": [
  {
    "tag" : "Name of the Disease",
    "patterns" : ["comma separated symptoms"],
    "responses" : ["Answer user will receive i.e the disease user might have"]
  },
  {
    "tag" : "Name of the Disease 2",
    "patterns" : ["comma separated symptoms"],
    "responses" : ["Answer user will receive i.e the disease user might have"]
  },
  ..............
  ]
}
After loading this JSON into our Python Code we will then divide them into Questions and answers, 

Questions – The symptoms said by the user.
Answers – the disease they might have.
We will be using three variables words, classes, and documents.

words – To store the Symptoms that the user will say.
classes – To store the disease they might be having.
document – This is used to store each pattern with its respective tags. This will be useful when we create the Bag of Words.
We will also create an object of the WordNetLemmatizer() class which we will use later. Also, we will tokenize each of the intents into words and classes using the nltk.word_tokenize() method which is useful to break longer symptoms into single words and helpful for the creation of a Bag of Words and also in the identification process of the disease. After successfully dividing that JSON into words and classes we will lemmatize each of the words and ignore the special characters/punctuations (mentioned inside ignore_letters) and convert that result as well as the classes into a set so that we don’t have the same symptom same answer many times.

The code for the above-explained step is given below.

lemmatizer = WordNetLemmatizer()
 
intents = json.loads(open("intents.json").read())
 
words = []
classes = []
documents = []
 
ignore_letters = ["?", "!", ".", ","]
 
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
 
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
 
words = sorted(set(words))
classes = sorted(set(classes))
Next, we will dump the words and classes using the pickle module and convert them into a pickle file which we will use during our chatbot demo. Pickle files are used to convert a Python object into a byte/stream and store it in some file or database for later use. The pickle module keeps track of objects it has already serialized so that later references to the same object will not be re-serialized, allowing for faster execution. Allows saving model in very little time.

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
We are storing them in write-binary (wb) format because we will need to open them and write in them in binary format.

Prepare the data to feed into the Neural Network
As we know that we need to feed a Neural Network with numerical values, so now we have to convert our data into numerical values. First, we are creating an empty list named dataset (user can give any name) and another list named template which will store only 0’s, it will act as a template and the number of zeros will be the same as of the element in classes. Now we will prepare the Bag of Words, and while doing so we will again Lemmatize the values.

dataset = []
template = [0]*len(classes)
 
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
 
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
 
    output_row = list(template)
    output_row[classes.index(document[1])] = 1
    dataset.append([bag, output_row])
 
random.shuffle(dataset)
dataset = np.array(dataset)
 
train_x = list(dataset[:, 0])
train_y = list(dataset[:, 1])
First creating the empty list dataset and template as mentioned above. After then we are running a for loop over each value in the documents list, then create an empty bag to store each combination, in the word_patterns variable firstly we are storing the 0th index values of the document i.e each symptom, again in the same variable we are lemmatizing each symptom and converting them into lower case, so later user doesn’t need to keep in mind about the emphasis on the Uppercase letters, then in the bag, we are appending 1 if the word is present in word_patterns, otherwise 0.

Next, copy the template list into the output_row variable and find the index of the disease name and put 1 in place of them and then append the bag as well as the output_row as a list. the dataset will now become a nested list.

Then we are just shuffling the dataset so that the values get shuffled (so the value remains balanced). After doing that we are converting it into a NumPy array. Lastly, we are splitting them into 0th-dimension values and 1st-dimension values (features and labels).

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
 
 
sgd = SGD(learning_rate=0.01, decay=1e-6,
          momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
 
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
 
model.save("chatbot_model.h5", hist)
print("Done!")
Entire Code of the Training Process
Now let’s look at the code that we will use to train our model. You can access the files required for the training purpose from here.

import random
import json
import pickle
 
import nltk
nltk.download('all')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
 
import numpy as np
 
lemmatizer = WordNetLemmatizer()
 
intents = json.loads(open("intents.json").read())
 
words = []
classes = []
documents = []
 
ignore_letters = ["?", "!", ".", ","]
 
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
 
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
 
words = sorted(set(words))
classes = sorted(set(classes))
 
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
 
dataset = []
template = [0]*len(classes)
 
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower())
                     for word in word_patterns]
 
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
 
    output_row = list(template)
    output_row[classes.index(document[1])] = 1
    dataset.append([bag, output_row])
 
random.shuffle(dataset)
dataset = np.array(dataset)
 
train_x = list(dataset[:, 0])
train_y = list(dataset[:, 1])
 
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
 
 
sgd = SGD(learning_rate=0.01,
          momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
 
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
 
model.save("chatbot_model.h5", hist)
print("Done!")
Output:

Epoch 196/200
111/111 [==============================] - 0s 1ms/step - loss: 0.6763 - accuracy: 0.7405
Epoch 197/200
111/111 [==============================] - 0s 1ms/step - loss: 0.6923 - accuracy: 0.7314
Epoch 198/200
111/111 [==============================] - 0s 1ms/step - loss: 0.6358 - accuracy: 0.7604
Epoch 199/200
111/111 [==============================] - 0s 1ms/step - loss: 0.7456 - accuracy: 0.7332
Epoch 200/200
111/111 [==============================] - 0s 1ms/step - loss: 0.6588 - accuracy: 0.7604
Done!
The number of Epochs might differ if you use a different number, so as the accuracy and the loss. This is how you can choose to train your own personal chatbot model using deep neural networks and train it on a huge corpus of data. But everyone does not have that many resources to train a model from scratch so, we will be using the pre-trained model to integrate into our chatbot.

Entering into the section where we make it speak
In the below code, a function has been implemented where it will use the pre-trained model to predict the disease based on the symptom of the patient. It will take the input as the symptom spoken by the patient into the chatbot and then using the chatbot_model the disease will be predicted and shown to the patient.

# This function will take the voice input
# converted into string as input and predict
# and return the result in both
# text as well as voice format
 
def calling_the_bot(txt):
    global res
    predict = predict_class(txt)
    res = get_response(predict, intents)
 
    engine.say("Found it. From our Database we found that" + res)
    # engine.say(res)
    engine.runAndWait()
    print("Your Symptom was  : ", text)
    print("Result found in our Database : ", res)
The above function is just used to call the predict_class() as well as the get_response() functions to get out the final output and print it in text mode as well as saying it loud using voice.

if __name__ == '__main__':
    print("Bot is Running")
 
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
 
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
 
    # Increase the rate of the bot according to need,
    # Faster the rate, faster it will speak, vice versa for slower.
 
    engine.setProperty('rate', 175)
 
    # Increase or decrease the bot's volume
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 1.0)
 
    voices = engine.getProperty('voices')
 
    engine.say(
        "Hello user, I am Bagley, your personal Talking Healthcare Chatbot.")
    engine.runAndWait()
 
    engine.say(
        "IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE SAY MALE.\
        OTHERWISE SAY FEMALE.")
    engine.runAndWait()
 
    # Asking for the MALE or FEMALE voice.
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.listen(source)
 
    audio = recognizer.recognize_google(audio)
 
    # If the user says Female then the bot will speak in female voice.
    if audio == "Female".lower():
        engine.setProperty('voice', voices[1].id)
        print("You have chosen to continue with Female Voice")
    else:
        engine.setProperty('voice', voices[0].id)
        print("You have chosen to continue with Male Voice")
 
    while True or final.lower() == 'True':
        with mic as symptom:
            print("Say Your Symptoms. The Bot is Listening")
            engine.say("You may tell me your symptoms now. I am listening")
            engine.runAndWait()
            try:
                recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
                symp = recognizer.listen(symptom)
                text = recognizer.recognize_google(symp)
                engine.say("You said {}".format(text))
                engine.runAndWait()
 
                engine.say(
                    "Scanning our database for your symptom. Please wait.")
                engine.runAndWait()
 
                time.sleep(1)
 
                # Calling the function by passing the voice
                # inputted symptoms converted into string
                calling_the_bot(text)
            except sr.UnknownValueError:
                engine.say(
                    "Sorry, Either your symptom is unclear to me \
                    or it is not present in our database. Please Try Again.")
                engine.runAndWait()
                print(
                    "Sorry, Either your symptom is unclear to me\
                    or it is not present in our database. Please Try Again.")
            finally:
                engine.say(
                    "If you want to continue please say True otherwise\
                    say False.")
                engine.runAndWait()
 
        with mic as ans:
            recognizer.adjust_for_ambient_noise(ans, duration=0.2)
            voice = recognizer.listen(ans)
            final = recognizer.recognize_google(voice)
 
        if final.lower() == 'no' or final.lower() == 'please exit':
            engine.say("Thank You. Shutting Down now.")
            engine.runAndWait()
            print("Bot has been stopped by the user")
            exit(0)
Firstly, we are taking the symptom as voice input from the user and then telling the bot to say it out loud so that the user can confirm he/she has said that, then calling the function calling_the_bot() which will predict and give us the final predicted result in both voice and text mode. We have enclosed that part in try-catch because If we ever encounter any error like the symptom is not there or our chatbot didn’t catch the symptom correctly, then our program will not crash, rather than that it will tell the user to retry and finally it will tell the user to say either True if they want to go again, this is why the second condition in while loop was introduced. If the user doesn’t want to continue, they can say False and the bot will stop.

Entire Healthcare Chatbot testing Code
While building such a complex application as a chatbot we need to integrate multiple functions which work in collaboration with each other. Below is a complete code that you can run with the dataset and files link which has been provided here to see your chatbot working.

import random
import json
import pickle
 
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
 
import numpy as np
import speech_recognition as sr
import pyttsx3
import time
 
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
 
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
 
 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
 
    return sentence_words
 
 
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
 
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
 
 
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
 
    ERROR_THRESHOLD = 0.25
 
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
 
    results.sort(key=lambda x: x[1], reverse=True)
 
    return_list = []
 
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
    return return_list
 
 
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
 
    result = ''
 
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
 
 
# This function will take the voice input converted
# into string as input and predict and return the result in both
# text as well as voice format.
def calling_the_bot(txt):
    global res
    predict = predict_class(txt)
    res = get_response(predict, intents)
 
    engine.say("Found it. From our Database we found that" + res)
    # engine.say(res)
    engine.runAndWait()
    print("Your Symptom was  : ", text)
    print("Result found in our Database : ", res)
 
 
if __name__ == '__main__':
    print("Bot is Running")
 
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
 
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
 
    # Increase the rate of the bot according to need,
    # Faster the rate, faster it will speak, vice versa for slower.
 
    engine.setProperty('rate', 175)
 
    # Increase or decrease the bot's volume
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 1.0)
 
    voices = engine.getProperty('voices')
 
    """User Might Skip the following Part till the start of While Loop"""
    engine.say(
        "Hello user, I am Bagley, your personal Talking Healthcare Chatbot.")
    engine.runAndWait()
 
    engine.say(
        "IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE\
        SAY MALE. OTHERWISE SAY FEMALE.")
    engine.runAndWait()
 
    # Asking for the MALE or FEMALE voice.
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.listen(source)
 
    audio = recognizer.recognize_google(audio)
 
    # If the user says Female then the bot will speak in female voice.
    if audio == "Female".lower():
        engine.setProperty('voice', voices[1].id)
        print("You have chosen to continue with Female Voice")
    else:
        engine.setProperty('voice', voices[0].id)
        print("You have chosen to continue with Male Voice")
 
    """User might skip till HERE"""
 
    while True or final.lower() == 'True':
        with mic as symptom:
            print("Say Your Symptoms. The Bot is Listening")
            engine.say("You may tell me your symptoms now. I am listening")
            engine.runAndWait()
            try:
                recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
                symp = recognizer.listen(symptom)
                text = recognizer.recognize_google(symp)
                engine.say("You said {}".format(text))
                engine.runAndWait()
 
                engine.say(
                    "Scanning our database for your symptom. Please wait.")
                engine.runAndWait()
 
                time.sleep(1)
 
                # Calling the function by passing the voice inputted
                # symptoms converted into string
                calling_the_bot(text)
            except sr.UnknownValueError:
                engine.say(
                    "Sorry, Either your symptom is unclear to me or it is\
                    not present in our database. Please Try Again.")
                engine.runAndWait()
                print(
                    "Sorry, Either your symptom is unclear to me or it is\
                    not present in our database. Please Try Again.")
            finally:
                engine.say(
                    "If you want to continue please say True otherwise say\
                    False.")
                engine.runAndWait()
 
        with mic as ans:
            recognizer.adjust_for_ambient_noise(ans, duration=0.2)
            voice = recognizer.listen(ans)
            final = recognizer.recognize_google(voice)
 
        if final.lower() == 'no' or final.lower() == 'please exit':
            engine.say("Thank You. Shutting Down now.")
            engine.runAndWait()
            print("Bot has been stopped by the user")
            exit(0)
