import DataParser

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

    #training the model 
    for epoch in range(10):
        parser.shuffle_training_data()
        for batch_start in range(0,len(train_data), batch_size):
            batch_data, batch_labels = parser.get_batch(batch_start:batch_start + batch_size)
           
