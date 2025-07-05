import csv
import numpy as np

class MNISTDataParser:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        
    def load_training_data(self):
        print("Loading training data...")
        
        with open(self.train_path, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader) 
            
            data_list = []
            labels_list = []
            
            for row in csv_reader:
                label = int(row[0])
                labels_list.append(label)
                pixel_values = [int(pixel) for pixel in row[1:]]
                data_list.append(pixel_values)

        self.train_data = np.array(data_list, dtype=np.float32)
        self.train_labels = np.array(labels_list, dtype=np.int32)
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Training labels shape: {self.train_labels.shape}")
        
        return self.train_data, self.train_labels
    
    def load_test_data(self):
        print("Loading test data...")
        
        with open(self.test_path, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            data_list = []
            
            for row in csv_reader:
                pixel_values = [int(pixel) for pixel in row]
                data_list.append(pixel_values)
        self.test_data = np.array(data_list, dtype=np.float32)
        
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.test_data

        def normalize_data(self):
        """Normalize pixel values to range [0, 1]"""
        if self.train_data is not None:
            self.train_data = self.train_data / 255.0
        if self.test_data is not None:
            self.test_data = self.test_data / 255.0
        print("Data normalized to range [0, 1]")
    
    def reshape_to_images(self):
        """Reshape flattened pixel data to 28x28 images"""
        if self.train_data is not None:
            self.train_data = self.train_data.reshape(-1, 28, 28)
        if self.test_data is not None:
            self.test_data = self.test_data.reshape(-1, 28, 28)
        print("Data reshaped to 28x28 images")
    
    def visualize_sample(self, index=0, dataset='train'):
        """Visualize a sample image from the dataset"""
        if dataset == 'train' and self.train_data is not None:
            if len(self.train_data.shape) == 2: 
                image = self.train_data[index].reshape(28, 28)
            else:
                image = self.train_data[index]
            
            plt.figure(figsize=(6, 6))
            plt.imshow(image, cmap='gray')
            plt.title(f'Training Sample {index}, Label: {self.train_labels[index]}')
            plt.axis('off')
            plt.show()
            
        elif dataset == 'test' and self.test_data is not None:
            if len(self.test_data.shape) == 2:  # Flattened data
                image = self.test_data[index].reshape(28, 28)
            else:  # Already reshaped
                image = self.test_data[index]
            
            plt.figure(figsize=(6, 6))
            plt.imshow(image, cmap='gray')
            plt.title(f'Test Sample {index}')
            plt.axis('off')
            plt.show()
    
    def create_one_hot_labels(self, num_classes=10):
        """Convert labels to one-hot encoding"""
        if self.train_labels is not None:
            one_hot = np.zeros((len(self.train_labels), num_classes))
            one_hot[np.arange(len(self.train_labels)), self.train_labels] = 1
            return one_hot
        return None
    
    def get_batch(self, batch_size, start_idx=0, dataset='train'):
        """Get a batch of data for training"""
        if dataset == 'train':
            end_idx = min(start_idx + batch_size, len(self.train_data))
            return (self.train_data[start_idx:end_idx], 
                   self.train_labels[start_idx:end_idx])
        elif dataset == 'test':
            end_idx = min(start_idx + batch_size, len(self.test_data))
            return self.test_data[start_idx:end_idx]
    
    def shuffle_training_data(self):
        """Shuffle training data and labels together"""
        if self.train_data is not None and self.train_labels is not None:
            indices = np.random.permutation(len(self.train_data))
            self.train_data = self.train_data[indices]
            self.train_labels = self.train_labels[indices]
            print("Training data shuffled")
