import DataParser
import numpy as np
import InceptionCNN

def main():
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

    #one hot encode the labels
    train_labels_one_hot = parser.one_hot_encode(train_labels)

    #initialize the model
    cnn = InceptionCNN(input_channels=1, num_classes=10)

    #hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001

    #training the model 
    for epoch in range(num_epochs):
        parser.shuffle_training_data()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0.0
        for batch_start in range(0,len(train_data), batch_size):
            batch_end = min(batch_start + batch_size, len(train_data))
            batch_data, batch_labels = parser.get_batch(batch_start:batch_start + batch_size)
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
        avg_accuracy = epoch_accuracy / num_batch 

        print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    print("done")
        
