import DataParser
import numpy as np
from InceptionCNN import InceptionCNN
import pickle
import time
from tqdm import tqdm

def save_model(model, filepath):
    """Save the trained model to a file"""
    model_data = {
        'conv1x1_weights': model.conv1x1_weights,
        'conv1x1_bias': model.conv1x1_bias,
        'conv1x1_3x3_weights': model.conv1x1_3x3_weights,
        'conv1x1_3x3_bias': model.conv1x1_3x3_bias,
        'conv3x3_weights': model.conv3x3_weights,
        'conv3x3_bias': model.conv3x3_bias,
        'conv1x1_5x5_weights': model.conv1x1_5x5_weights,
        'conv1x1_5x5_bias': model.conv1x1_5x5_bias,
        'conv5x5_weights': model.conv5x5_weights,
        'conv5x5_bias': model.conv5x5_bias,
        'conv1x1_pool_weights': model.conv1x1_pool_weights,
        'conv1x1_pool_bias': model.conv1x1_pool_bias,
        'fc_weights': model.fc_weights,
        'fc_bias': model.fc_bias,
        'input_channels': model.input_channels,
        'num_classes': model.num_classes
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")

def main():
    print("Starting the CNN training process...")
    #bruh init
    parser = DataParser.MNISTDataParser(
        train_path=r"c:\Users\Victus\Desktop\AI- digit recognistion\CNN, handwirtten digit recgonition\test and training set\train.csv",
        test_path=r"c:\Users\Victus\Desktop\AI- digit recognistion\CNN, handwirtten digit recgonition\test and training set\test.csv"
    )
    #load the data
    train_data, train_labels = parser.load_training_data()
    test_data = parser.load_test_data()

    #normalize the data 
    parser.normalize_data()
    parser.reshape_to_images() #this reshapes (N, 784) -> (N, 28, 28)
    train_data = parser.train_data
    test_data = parser.test_data 
    train_data = train_data[:, np.newaxis, :, :]
    test_data = test_data[:, np.newaxis, :, :]
    
    
    print(f"Final train_data shape: {train_data.shape}") 
    print(f"Final test_data shape: {test_data.shape}")

    #one hot encode the labels
    train_labels_one_hot = parser.create_one_hot_labels()

    #initialize the model
    cnn = InceptionCNN(input_channels=1, num_classes=10)

    #hyperparameters
    batch_size = 16
    num_epochs = 3
    learning_rate = 0.01

    training_history = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'training_time': []
    }
    start_time = time.time()
    
    #training the model 
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        parser.shuffle_training_data()

        train_data_shuffled = parser.train_data[:, np.newaxis, :, :] 
        train_labels_one_hot_shuffled = parser.create_one_hot_labels()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        total_batches = len(train_data_shuffled) // batch_size
        
        batch_indices = list(range(0, len(train_data_shuffled), batch_size))
        batch_progress = tqdm(batch_indices, desc=f"Epoch {epoch + 1}", total=total_batches, unit="batch", leave=False)

        for batch_start in batch_progress:
            print(f"Processing batch starting at index {batch_start}...")
            batch_end = min(batch_start + batch_size, len(train_data_shuffled))
            batch_data = train_data_shuffled[batch_start:batch_end]
            batch_one_hot = train_labels_one_hot_shuffled[batch_start:batch_end]

            loss, acc = cnn.train_step(batch_data, batch_one_hot, learning_rate)
            
            epoch_loss += loss
            epoch_accuracy += acc
            num_batches += 1

            #Update progress bar
            batch_progress.set_postfix({
                'Loss': f'{loss:.4f}',
                'Acc': f'{acc:.2%}',
                'Batch': f'{num_batches}'
            })

        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        epoch_time = time.time() - epoch_start

        training_history['epoch_losses'].append(avg_loss)
        training_history['epoch_accuracies'].append(avg_accuracy)
        training_history['training_time'].append(epoch_time)

        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

        if epoch > 0 and avg_loss > training_history['epoch_losses'][-2]:
            learning_rate *= 0.9
            print(f"Reduced learning rate to {learning_rate:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Final accuracy: {training_history['epoch_accuracies'][-1]:.4f}")

    #Save the trained model
    model_path = "trained_inception_cnn.pkl"
    save_model(cnn, model_path)

    with open("training_history.pkl", 'wb') as f:
        pickle.dump(training_history, f)
    print("Training history saved to training_history.pkl")

    print("\nTesting the model on test data...")
    test_sample = test_data[:5]
    test_predictions = cnn.forward(test_sample)
    predicted_digits = np.argmax(test_predictions, axis=1)
    print(f"Predicted digits: {predicted_digits}")
    print(f"Prediction probabilities:")
    for i, pred in enumerate(test_predictions):
        print(f"  Sample {i}: {pred}")        
if __name__ == "__main__":
    main()