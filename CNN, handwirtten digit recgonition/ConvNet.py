import DataParser
import numpy as np
from InceptionCNN import InceptionCNN

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
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001

    #training the model 
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        parser.shuffle_training_data()

        train_data_shuffled = parser.train_data[:, np.newaxis, :, :] 
        train_labels_shuffled = parser.train_labels
        train_labels_one_hot_shuffled = parser.create_one_hot_labels()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        for batch_start in range(0,len(train_data_shuffled), batch_size):
            print(f"Processing batch starting at index {batch_start}...")
            batch_end = min(batch_start + batch_size, len(train_data_shuffled))
            batch_data = train_data_shuffled[batch_start:batch_end]
            batch_one_hot = train_labels_one_hot[batch_start:batch_end]

            prediction = cnn.forward(batch_data)

            loss = cnn.cross_entropy_loss(prediction, batch_one_hot)
            acc = cnn.accuracy(prediction, batch_one_hot)

            epoch_loss += loss
            epoch_accuracy += acc
            num_batches += 1

            #progress every 50 batches
            if num_batches % 50 == 0:
                print(f"  Batch {num_batches}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    print("\nDone training the model!")
    print("\nTesting on a small sample...")
    test_sample = test_data[:5]
    test_predictions = cnn.forward(test_sample)
    predicted_digits = np.argmax(test_predictions, axis=1)
    print(f"Predicted digits: {predicted_digits}")
    print(f"Prediction probabilities:")
    for i, pred in enumerate(test_predictions):
        print(f"  Sample {i}: {pred}")        
if __name__ == "__main__":
    main()