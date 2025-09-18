from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import pickle
from keras.applications import ResNet50 #importing resnet50 class
from keras.applications import ResNet101#importing resnet101 class
from keras.applications import VGG16 #importing VGG16 class
from keras.applications import InceptionV3 #importing Inception GoogleNet class
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import pandas as pd
from keras import Model, layers
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from keras.models import load_model
import webbrowser

main = tkinter.Tk()
main.title("Breast Cancer Diagnosis on Pathological Images Data Augmentation Method: Cycle GAN") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y, X_train, X_test, y_train, y_test, vgg_classifier
labels = ['Normal', 'Cancer']
precision = []
recall = []
fscore = []
accuracy = []

def upload():
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
def Preprocessing():
    global filename, X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        X = np.load("model/X.txt.npy")
        Y = np.load("model/Y.txt.npy")       
    else:
        X = []
        Y = []
        path = filename
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                img = cv2.imread(root+"/"+directory[j]) #read image from dataset directory
                img = cv2.resize(img, (75, 75)) #resize image
                im2arr = np.array(img)
                im2arr = im2arr.reshape(75, 75, 3) #image as 3 colour format
                X.append(im2arr) #add images to array
                label = 0
                if name == "Cancer":
                    label = 1
                Y.append(label) #add class label to Y variable
                print(name+" "+str(label))
        X = np.asarray(X) #convert array images to numpy array
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.insert(END,"Total images found in dataset: "+str(X.shape[0])+"\n\n")
    X = X.astype('float32')
    X = X/255 #normalize image
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle images data
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and tesrt
    text.insert(END,"Dataset train & test split as 80% dataset for training and 20% for testing\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Training Size (20%): "+str(X_test.shape[0])+"\n")

#function to calculate accuracy and other metrics by comparing original label and predicted label
def calculateMetrics(algorithm, predict, testY):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100  
    conf_matrix = confusion_matrix(testY, predict) 
    se = (conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])) * 100
    sp = (conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])) * 100
    text.insert(END,algorithm+' Accuracy    : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FScore      : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    
        

def trainResnet50():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    #train Resnet50 algorithms
    #defining RESNET50 object and then adding layers for imagenet with CNN and max pooling filter layers
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False
    resnet_classifier = Sequential()
    resnet_classifier.add(resnet)
    resnet_classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnet_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    resnet_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_classifier.add(Flatten())
    resnet_classifier.add(Dense(output_dim = 256, activation = 'relu'))
    resnet_classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(resnet_classifier.summary())
    resnet_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet50_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet50_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet_classifier.fit(X_train, y_train, batch_size=64, epochs=10, shuffle=True, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/resnet50_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        resnet_classifier = load_model("model/resnet50_weights.hdf5")
    predict = resnet_classifier.predict(X_test) #perform prediction on test data
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Resnet50", predict, testY)  #call function to calculate metrics    
    

def trainResnet101():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    #defining RESNET101 object and then adding layers for imagenet with CNN and max pooling filter layers
    resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False
    resnet101_classifier = Sequential()
    resnet101_classifier.add(resnet)
    resnet101_classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnet101_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    resnet101_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    resnet101_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    resnet101_classifier.add(Flatten())
    resnet101_classifier.add(Dense(output_dim = 256, activation = 'relu'))
    resnet101_classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(resnet101_classifier.summary())
    resnet101_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet101_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet101_weights.hdf5', verbose = 1, save_best_only = True)
        hist = classifier.fit(X_train, y_train, batch_size=64, epochs=10, shuffle=True, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/resnet101_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        resnet101_classifier.load_weights("model/resnet101_weights.hdf5")
    #prediction on test data using resnet101
    predict = resnet101_classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Resnet101", predict, testY)

def trainGooglenet():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    #training with GoogleNet group algorithm called inceptionv3
    inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in inception.layers:
        layer.trainable = False
    inception_classifier = Sequential()
    inception_classifier.add(inception)
    inception_classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    inception_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    inception_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    inception_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    inception_classifier.add(Flatten())
    inception_classifier.add(Dense(output_dim = 256, activation = 'relu'))
    inception_classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(inception_classifier.summary())
    inception_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/inception_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/inception_weights.hdf5', verbose = 1, save_best_only = True)
        hist = inception_classifier.fit(X_train, y_train, batch_size=64, epochs=10, shuffle=True, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/inception_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        inception_classifier.load_weights("model/inception_weights.hdf5")
    #prediction on test data using GoogleNet
    predict = inception_classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("GoogleNet", predict, testY)     

def trainVGG():
    global X_train, X_test, y_train, y_test, vgg_classifier
    global accuracy, precision, recall, fscore
    #defining vgg16 object and then adding layers for imagenet with CNN and max pooling filter layers
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in vgg.layers:
        layer.trainable = False
    vgg_classifier = Sequential()
    vgg_classifier.add(vgg)
    vgg_classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vgg_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    vgg_classifier.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_classifier.add(Flatten())
    vgg_classifier.add(Dense(output_dim = 256, activation = 'relu'))
    vgg_classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(vgg_classifier.summary())
    vgg_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg_classifier.fit(X_train, y_train, batch_size=64, epochs=10, shuffle=True, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        vgg_classifier.load_weights("model/vgg_weights.hdf5")
    #prediction on test data using VGG16
    predict = vgg_classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("VGG16", predict, testY)

def trainAlexnet():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    #training with Alexnet algortihm
    #defining layers of alexnet model
    import keras
    alexnet_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    #compiling alexnet model
    alexnet_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/alexnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/alexnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = alexnet_model.fit(X_train, y_train, batch_size=64, epochs=10, shuffle=True, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/alexnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        alexnet_model.load_weights("model/alexnet_weights.hdf5")
    print(alexnet_model.summary())
    #prediction on test data using Alexnet
    predict = alexnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Alexnet", predict, testY)

def comparisonGraph():
    text.delete('1.0', END)
    output = '<table border=1 align=center>'
    output+= '<tr><th>Dataset Name</th><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th><th>Sensitivity</th><th>Specificity</th></tr>'
    output+='<tr><td>Breast Cancer Dataset</td><td>Resnet50</td><td>'+str(accuracy[0])+'</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td></tr>'
    output+='<tr><td>Breast Cancer Dataset</td><td>Resnet101</td><td>'+str(accuracy[1])+'</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td></tr>'
    output+='<tr><td>Breast Cancer Dataset</td><td>GoogleNet</td><td>'+str(accuracy[2])+'</td><td>'+str(precision[2])+'</td><td>'+str(recall[2])+'</td><td>'+str(fscore[2])+'</td></tr>'
    output+='<tr><td>Breast Cancer Dataset</td><td>VGG16</td><td>'+str(accuracy[3])+'</td><td>'+str(precision[3])+'</td><td>'+str(recall[3])+'</td><td>'+str(fscore[3])+'</td></tr>'
    output+='<tr><td>Breast Cancer Dataset</td><td>Alexnet</td><td>'+str(accuracy[4])+'</td><td>'+str(precision[4])+'</td><td>'+str(recall[4])+'</td><td>'+str(fscore[4])+'</td></tr>'
    
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)   
    df = pd.DataFrame([['Resnet50','FScore',fscore[0]],['Resnet50','Recall',recall[0]],['Resnet50','Precision',precision[0]],['Resnet50','Accuracy',accuracies[0]],
                   ['Resnet101','FScore',fscore[1]],['Resnet101','Recall',recall[1]],['Resnet101','Precision',precision[1]],['Resnet101','Accuracy',accuracies[1]], 
                   ['GoogleNet','FScore',fscore[2]],['GoogleNet','Recall',recall[2]],['GoogleNet','Precision',precision[2]],['GoogleNet','Accuracy',accuracies[2]],
                   ['VGG16','FScore',fscore[3]],['VGG16','Recall',recall[3]],['VGG16','Precision',precision[3]],['VGG16','Accuracy',accuracies[3]], 
                   ['Alexnet','FScore',fscore[4]],['Alexnet','Recall',recall[4]],['Alexnet','Precision',precision[4]],['Alexnet','Accuracy',accuracies[4]],   
    ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()


def predict():
    text.delete('1.0', END)
    global vgg_classifier
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (75,75))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,75,75,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = vgg_classifier.predict(img)
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Predicted As : '+labels[predict], img)
    cv2.waitKey(0)


font = ('times', 16, 'bold')
title = Label(main, text='Breast Cancer Diagnosis on Pathological Images Data Augmentation Method: Cycle GAN')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Breast Cancer Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Preprocess Dataset", command=Preprocessing)
preButton.place(x=350,y=100)
preButton.config(font=font1) 

trainButton = Button(main, text="Train Resnet50 Algorithm", command=trainResnet50)
trainButton.place(x=670,y=100)
trainButton.config(font=font1)

resnet101Button = Button(main, text="Train Resnet101 Algorithm", command=trainResnet101)
resnet101Button.place(x=50,y=150)
resnet101Button.config(font=font1)

googlenetButton = Button(main, text="Train GoogleNet Algorithm", command=trainGooglenet)
googlenetButton.place(x=350,y=150)
googlenetButton.config(font=font1)

vggButton = Button(main, text="Train VGG16 Algorithm", command=trainVGG)
vggButton.place(x=670,y=150)
vggButton.config(font=font1)

alexnetButton = Button(main, text="Train Alexnet Algorithm", command=trainAlexnet)
alexnetButton.place(x=50,y=200)
alexnetButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=trainAlexnet)
graphButton.place(x=350,y=200)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Cancer from Test Image", command=predict)
predictButton.place(x=670,y=200)
predictButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
