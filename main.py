# Siamese Neural Network for Face Recognition
# This program uses deep learning to verify if two face images are of the same person

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random

# configuration stuff
IMG_HEIGHT = 105  # height of images
IMG_WIDTH = 105  # width of images
IMG_CHANNELS = 3  # RGB channels
BATCH_SIZE = 32  # how many images to process at once
EPOCHS = 20  # number of times to train through the dataset
LFW_PATH = 'lfw/lfw-deepfunneled'  # path to the dataset folder

# setup GPU so it doesn't use all the memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# function to load all the images from the dataset
def load_dataset(dataset_path):
    people = {}  # dictionary to store images by person
    
    # go through each folder (each person has their own folder)
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            images = []  # list to store image paths for this person
            # get all the image files for this person
            for img_file in os.listdir(person_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    images.append(img_path)
            
            # only use people who have at least 2 images (need pairs for training)
            if len(images) >= 2:
                people[person_name] = images
    
    print(f"Loaded {len(people)} people")
    print(f"Total images: {sum(len(imgs) for imgs in people.values())}")
    
    return people

# function to load a single image and preprocess it
def load_image(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = cv2.imread(img_path)  # read the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    img = cv2.resize(img, target_size)  # resize to the size we want
    img = img.astype('float32') / 255.0  # normalize pixel values to 0-1
    return img

# create pairs of images for training
# positive pairs = same person, negative pairs = different people
def create_pairs(people_dict, num_pairs=2000):
    pairs = []  # store the image pairs
    labels = []  # store the labels (1 = same person, 0 = different)
    people_names = list(people_dict.keys())
    
    # create positive pairs (same person)
    for _ in range(num_pairs // 2):
        person = random.choice(people_names)  # pick a random person
        if len(people_dict[person]) < 2:
            continue
        # pick two different images of the same person
        img1_path, img2_path = random.sample(people_dict[person], 2)
        pairs.append([img1_path, img2_path])
        labels.append(1)  # label as same person
    
    # create negative pairs (different people)
    for _ in range(num_pairs // 2):
        # pick two different people
        person1, person2 = random.sample(people_names, 2)
        img1_path = random.choice(people_dict[person1])
        img2_path = random.choice(people_dict[person2])
        pairs.append([img1_path, img2_path])
        labels.append(0)  # label as different people
    
    return np.array(pairs), np.array(labels)

# build the base CNN that creates embeddings
# this network is shared between both images (siamese means "twin")
def build_base_network(input_shape):
    input_layer = Input(shape=input_shape)
    
    # first convolutional block
    x = Conv2D(64, (10, 10), activation='relu')(input_layer)  # 64 filters, 10x10 size
    x = MaxPooling2D((2, 2))(x)  # downsample by 2
    
    # second convolutional block
    x = Conv2D(128, (7, 7), activation='relu')(x)  # 128 filters, 7x7 size
    x = MaxPooling2D((2, 2))(x)
    
    # third convolutional block
    x = Conv2D(128, (4, 4), activation='relu')(x)  # 128 filters, 4x4 size
    x = MaxPooling2D((2, 2))(x)
    
    # fourth convolutional block
    x = Conv2D(256, (4, 4), activation='relu')(x)  # 256 filters, 4x4 size
    
    # flatten the output and create dense layer
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)  # 4096 dimensional embedding
    
    return Model(input_layer, x, name='base_network')

# calculate the distance between two embeddings
@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects  # the two embeddings
    # calculate euclidean distance
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

# build the full siamese network
def build_siamese_model(input_shape):
    # two inputs (one for each image in the pair)
    input_a = Input(shape=input_shape, name='input_a')
    input_b = Input(shape=input_shape, name='input_b')
    
    # create the shared base network
    base_network = build_base_network(input_shape)
    
    # pass both images through the same network to get embeddings
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)
    
    # calculate distance between the embeddings
    distance = Lambda(euclidean_distance, output_shape=(1,))([embedding_a, embedding_b])
    
    # final classification layer - outputs probability they're the same person
    output = Dense(1, activation='sigmoid')(distance)
    
    # create the full model
    model = Model(inputs=[input_a, input_b], outputs=output, name='siamese_network')
    
    return model

# training function
def train_model(people_dict, epochs=EPOCHS, batch_size=BATCH_SIZE):
    
    print("\n" + "="*60)
    print("CREATING PAIRS")
    print("="*60)
    
    # create training and validation pairs
    train_pairs, train_labels = create_pairs(people_dict, num_pairs=2000)
    val_pairs, val_labels = create_pairs(people_dict, num_pairs=400)
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    # build the model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = build_siamese_model(input_shape)
    
    # compile with adam optimizer and binary crossentropy loss
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()  # print model architecture
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    # store training history
    history_data = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    # training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # shuffle training data each epoch
        indices = np.arange(len(train_pairs))
        np.random.shuffle(indices)
        train_pairs = train_pairs[indices]
        train_labels = train_labels[indices]
        
        # lists to store metrics for this epoch
        train_loss = []
        train_acc = []
        
        # train on batches
        for i in range(0, len(train_pairs), batch_size):
            batch_pairs = train_pairs[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            # load the images for this batch
            batch_img1 = np.array([load_image(pair[0]) for pair in batch_pairs])
            batch_img2 = np.array([load_image(pair[1]) for pair in batch_pairs])
            
            # train on this batch
            metrics = model.train_on_batch([batch_img1, batch_img2], batch_labels)
            train_loss.append(metrics[0])
            train_acc.append(metrics[1])
        
        # validation on batches
        val_loss = []
        val_acc = []
        
        for i in range(0, len(val_pairs), batch_size):
            batch_pairs = val_pairs[i:i+batch_size]
            batch_labels = val_labels[i:i+batch_size]
            
            # load validation images
            batch_img1 = np.array([load_image(pair[0]) for pair in batch_pairs])
            batch_img2 = np.array([load_image(pair[1]) for pair in batch_pairs])
            
            # evaluate on this batch
            metrics = model.test_on_batch([batch_img1, batch_img2], batch_labels)
            val_loss.append(metrics[0])
            val_acc.append(metrics[1])
        
        # calculate average metrics for the epoch
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_val_loss = np.mean(val_loss)
        epoch_val_acc = np.mean(val_acc)
        
        # save to history
        history_data['loss'].append(epoch_train_loss)
        history_data['accuracy'].append(epoch_train_acc)
        history_data['val_loss'].append(epoch_val_loss)
        history_data['val_accuracy'].append(epoch_val_acc)
        
        # print metrics
        print(f"Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")
        
        # save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save(f'siamese_model_epoch_{epoch+1}.h5')
            print(f"Saved checkpoint")
    
    # save final model
    model.save('siamese_model_final.h5')
    print("\nTraining complete! Model saved as 'siamese_model_final.h5'")
    
    # plot the training history
    plt.figure(figsize=(12, 4))
    
    # plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_data['accuracy'], label='Train Accuracy')
    plt.plot(history_data['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history_data['loss'], label='Train Loss')
    plt.plot(history_data['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'")
    plt.show()
    
    return model

# function to verify if two face images are the same person
def verify_face(model, img1_path, img2_path, threshold=0.5):
    # load both images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    # add batch dimension
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    
    # get prediction
    prediction = model.predict([img1, img2], verbose=0)[0][0]
    
    # determine if same person based on threshold
    is_same = prediction > threshold
    confidence = prediction if is_same else (1 - prediction)
    
    return is_same, confidence

# real-time verification using webcam
def realtime_verification(model_path='siamese_model_final.h5'):
    # check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    # load the trained model with custom objects
    print("Loading model...")
    model = keras.models.load_model(model_path, custom_objects={'euclidean_distance': euclidean_distance})
    
    # get reference image to compare against
    ref_image_path = input("Enter path to reference image: ")
    if not os.path.exists(ref_image_path):
        print("Reference image not found!")
        return
    
    # load and prepare reference image
    ref_img = load_image(ref_image_path)
    ref_img = np.expand_dims(ref_img, axis=0)
    
    # load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # start webcam
    cap = cv2.VideoCapture(0)
    
    print("\nPress 'q' to quit")
    
    # main loop
    while True:
        ret, frame = cap.read()  # read frame from webcam
        if not ret:
            break
        
        # convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # process each detected face
        for (x, y, w, h) in faces:
            # extract face region
            face_roi = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IMG_HEIGHT, IMG_WIDTH))
            face_normalized = face_resized.astype('float32') / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # compare with reference image
            prediction = model.predict([ref_img, face_batch], verbose=0)[0][0]
            is_match = prediction > 0.5
            
            # draw rectangle and label
            color = (0, 255, 0) if is_match else (0, 0, 255)  # green if match, red if not
            label = f"Match: {prediction:.2f}" if is_match else f"No Match: {1-prediction:.2f}"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # show the frame
        cv2.imshow('Face Verification', frame)
        
        # quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()

# main program
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIAMESE NETWORK - FACIAL RECOGNITION")
    print("="*60)
    
    # check if dataset exists
    if not os.path.exists(LFW_PATH):
        print(f"\nERROR: Dataset not found at '{LFW_PATH}'")
        print("Please update LFW_PATH in the configuration section")
        exit()
    
    # load the dataset
    print(f"\nLoading dataset from: {LFW_PATH}")
    people_dict = load_dataset(LFW_PATH)
    
    # make sure we got data
    if len(people_dict) == 0:
        print("No data found!")
        exit()
    
    # ask if user wants to train
    print("\nStart training? (y/n): ", end="")
    if input().lower() == 'y':
        model = train_model(people_dict, epochs=EPOCHS)
    
    # ask if user wants to run real-time verification
    print("\nRun real-time verification? (y/n): ", end="")
    if input().lower() == 'y':
        realtime_verification()