import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def identity_block(X, f, filters, initializer=random_uniform):
    F1, F2, F3 = filters
    
    X_shortcut = X
 
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X) 
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X) 
  
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X) 

    return X

def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X) 
    X = BatchNormalization(axis = 3)(X) 
    X = Activation('relu')(X) 

    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X) 
    X = BatchNormalization(axis = 3)(X) 

    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s,s), padding = 'valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (64, 64, 3), classes = 6, training=False):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
 
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)

    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)

    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)

    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = GlobalAveragePooling2D()(X)

    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X)

    return model

def ResNet50_MNIST(input_shape=(28, 28, 1), classes=10, training=False):
    """ResNet50 adapted specifically for MNIST 28x28 grayscale images"""
    X_input = Input(input_shape)

    X = ZeroPadding2D((1, 1))(X_input) 
   
    X = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)  


    X = convolutional_block(X, f=3, filters=[16, 16, 64], s=1)
    X = identity_block(X, 3, [16, 16, 64])

 
    X = convolutional_block(X, f=3, filters=[32, 32, 128], s=2)
    X = identity_block(X, 3, [32, 32, 128])


    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=2)
    X = identity_block(X, 3, [64, 64, 256])

    X = GlobalAveragePooling2D()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet50_MNIST')
    return model

def load_and_preprocess_data():
    print("Loading training data...")
    train_df = pd.read_csv(r"c:\Users\Victus\Desktop\AI- digit recognistion\CNN, handwirtten digit recgonition\test and training set\train.csv")
    
    print("Loading test data...")
    test_df = pd.read_csv(r"c:\Users\Victus\Desktop\AI- digit recognistion\CNN, handwirtten digit recgonition\test and training set\test.csv")
    
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values
    
    X_test = test_df.values
    
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_test

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

def create_prediction_wrapper(model_path='resnet50_mnist_model.h5'):
    """Create a wrapper class that mimics InceptionCNN interface for DigitDrawingApp"""
    class ResNetWrapper:
        def __init__(self, model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"ResNet model loaded from {model_path}")
        
        def predict_single(self, image_array):
            image_batch = image_array.reshape(1, 28, 28, 1)
    
            predictions = self.model.predict(image_batch, verbose=0)[0]

            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit]
            
            return predicted_digit, confidence, predictions
    
    return ResNetWrapper(model_path)

def main():
    print("Starting ResNet50 MNIST Training...")
    
    X_train, y_train, X_test = load_and_preprocess_data()

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    model = ResNet50_MNIST(input_shape=(28, 28, 1), classes=10)

    print(model.summary())
 
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    print("Starting training...")
    history = model.fit(
        X_train_split, y_train_split,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    plot_training_history(history)

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    model.save('resnet50_mnist_model.h5')
    print("Model saved as 'resnet50_mnist_model.h5'")

    wrapper = create_prediction_wrapper('resnet50_mnist_model.h5')
    print("Model wrapper created successfully!")
  
    test_image = X_val[0]  
    pred_digit, confidence, predictions = wrapper.predict_single(test_image)
    print(f"Test prediction: Digit {pred_digit} with {confidence:.2%} confidence")

    print("Making predictions on test set...")
    test_predictions = model.predict(X_test)
    predicted_labels = np.argmax(test_predictions, axis=1)

    submission_df = pd.DataFrame({
        'ImageId': range(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })
    submission_df.to_csv('resnet_submission.csv', index=False)
    print("Predictions saved to 'resnet_submission.csv'")
    
    return model, history

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    model, history = main()

