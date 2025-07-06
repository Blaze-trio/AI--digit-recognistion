import numpy as np
import pickle

class InceptionCNN:
    def __init__(self, input_channels=1, num_classes=10):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.initialize_weights()
    
    def xavier_uniform(self, fan_in, fan_out, shape):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def xavier_normal(self, fan_in, fan_out, shape):
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)
    
    def calculate_conv_fan_in_out(self, input_channels, output_channels, kernel_height, kernel_width):
        fan_in = input_channels * kernel_height * kernel_width
        fan_out = output_channels * kernel_height * kernel_width
        return fan_in, fan_out

    def initialize_weights(self):
        print("\nInitializing weights for InceptionCNN")
        # 1X1 convolution layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 32, 1, 1)
        self.conv1x1_weights = self.xavier_uniform(fan_in,fan_out, (32,self.input_channels, 1, 1))
        self.conv1x1_bias = np.zeros((32,))

        # 3X3 convolution layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels,16, 1, 1)
        self.conv1x1_3x3_weights = self.xavier_uniform(fan_in, fan_out, (16, self.input_channels, 1, 1))
        self.conv1x1_3x3_bias = np.zeros((16,))
        fan_in, fan_out = self.calculate_conv_fan_in_out(16, 32, 3, 3)
        self.conv3x3_weights = self.xavier_uniform(fan_in, fan_out, (32, 16, 3, 3))
        self.conv3x3_bias = np.zeros((32,))

        # 5X5 convolution layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 8, 1, 1)
        self.conv1x1_5x5_weights = self.xavier_uniform(fan_in, fan_out, (8, self.input_channels, 1, 1))
        self.conv1x1_5x5_bias = np.zeros((8,))
        fan_in, fan_out = self.calculate_conv_fan_in_out(8, 32, 5, 5)
        self.conv5x5_weights = self.xavier_uniform(fan_in, fan_out, (32, 8, 5, 5))
        self.conv5x5_bias = np.zeros((32,))

        # max pooling layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 32, 3, 3)
        self.conv1x1_pool_weights = self.xavier_uniform(fan_in, fan_out, (32, self.input_channels, 1, 1))
        self.conv1x1_pool_bias = np.zeros((32,))

        # fully connected layer
        fc_input_size = 128 * 28 * 28 
        fc_output_size = self.num_classes

        fan_in, fan_out = fc_input_size, fc_output_size
        self.fc_weights = self.xavier_uniform(fan_in, fan_out, (fc_input_size, fc_output_size))
        self.fc_bias = np.zeros((fc_output_size,))
    
    def pad_input(self, inputData, padding):
        print("\npad_input")
        batchSize, channels, height, width = inputData.shape
        padded = np.zeros((batchSize, channels, height + 2*padding, width + 2*padding))
        padded[:, :, padding:height+padding, padding:width+padding] = inputData
        return padded
    
    def conv2d(self, inputData, weights, bias, stride=1, padding=0):
        print("\nconv2d")
        if padding > 0:
            inputData = self.pad_input(inputData, padding)

        batchSize, inChannels, inHeight, inWidth = inputData.shape
        outChannels, _, filterHeight, filterWidth = weights.shape

        outHeight = int((inHeight - filterHeight) / stride) + 1
        outWidth = int((inWidth - filterWidth) / stride) + 1

        output = np.zeros((batchSize, outChannels, outHeight, outWidth))

        for b in range(batchSize):
            for c in range(outChannels):
                for h in range(outHeight):
                    for w in range(outWidth):
                        hStart = h * stride
                        hEnd = hStart + filterHeight
                        wStart = w * stride
                        wEnd = wStart + filterWidth
                        inputSlice = inputData[b, :, hStart:hEnd, wStart:wEnd]
                        output[b, c, h, w] = np.sum(inputSlice * weights[c]) + bias[c]
        return output

    def relu(self, x):
        return np.maximum(0, x)
    
    def max_pool2d(self, inputData, poolSize=2, stride=2, padding=0):
        print("\nmax_pool2d")
        if padding > 0:
            inputData = self.pad_input(inputData, padding)

        batchSize, channels, inHeight, inWidth = inputData.shape
        outHeight = int((inHeight - poolSize) / stride) + 1
        outWidth = int((inWidth - poolSize) / stride) + 1

        output = np.zeros((batchSize, channels, outHeight, outWidth))

        for b in range(batchSize):
            for c in range(channels):
                for h in range(outHeight):
                    for w in range(outWidth):
                        hStart = h * stride
                        hEnd = hStart + poolSize
                        wStart = w * stride
                        wEnd = wStart + poolSize
                        inputSlice = inputData[b, c, hStart:hEnd, wStart:wEnd]
                        output[b, c, h, w] = np.max(inputSlice)
        return output

    def inception_block(self, inputData):
        print("\ninception_block")
        # 1x1 convolution
        conv1x1 = self.conv2d(inputData, self.conv1x1_weights, self.conv1x1_bias)
        conv1x1 = self.relu(conv1x1)

        # 3x3 convolution
        conv3x3 = self.conv2d(inputData, self.conv1x1_3x3_weights, self.conv1x1_3x3_bias)
        conv3x3 = self.relu(conv3x3)
        padding = int((3 - 1) / 2)#same padding
        conv3x3 = self.conv2d(conv3x3, self.conv3x3_weights, self.conv3x3_bias, padding=1)
        conv3x3 = self.relu(conv3x3)

        # 5x5 convolution
        conv5x5 = self.conv2d(inputData, self.conv1x1_5x5_weights, self.conv1x1_5x5_bias)
        conv5x5 = self.relu(conv5x5)
        padding = int((5 - 1) / 2)
        conv5x5 = self.conv2d(conv5x5, self.conv5x5_weights, self.conv5x5_bias, padding=2)
        conv5x5 = self.relu(conv5x5)

        # pooling
        padding = int((3 - 1) / 2)
        pool = self.max_pool2d(inputData, poolSize=3, stride=1, padding=1)
        pool = self.conv2d(pool, self.conv1x1_pool_weights, self.conv1x1_pool_bias)
        pool = self.relu(pool)

        output = np.concatenate((conv1x1, conv3x3, conv5x5, pool), axis=1)
        return output
    
    def softmax(self, x):
        print("\nsoftmax")
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, inputData):
        print("\nforward")
        inputData = self.inception_block(inputData)

        batchSize, channels, height, width = inputData.shape
        inputData = inputData.reshape(batchSize, -1)

        inputData = np.dot(inputData, self.fc_weights) + self.fc_bias
        inputData = self.relu(inputData)

        output = self.softmax(inputData)
        return output

    def cross_entropy_loss(self, predictions, labels):
        print("\ncross_entropy_loss")
        batchSize = predictions.shape[0]
        
        epsilon = 1e-12 
        predictions = np.clip(predictions, epsilon, 1. - epsilon)

        loss = -np.sum(labels * np.log(predictions)) / batchSize
        return loss

    def accuracy(self, predictions, labels):
        print("\naccuracy")
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy
    
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        #Create new instance
        model = cls(
            input_channels=model_data['input_channels'],
            num_classes=model_data['num_classes']
        )
        
        #Load weights
        model.conv1x1_weights = model_data['conv1x1_weights']
        model.conv1x1_bias = model_data['conv1x1_bias']
        model.conv1x1_3x3_weights = model_data['conv1x1_3x3_weights']
        model.conv1x1_3x3_bias = model_data['conv1x1_3x3_bias']
        model.conv3x3_weights = model_data['conv3x3_weights']
        model.conv3x3_bias = model_data['conv3x3_bias']
        model.conv1x1_5x5_weights = model_data['conv1x1_5x5_weights']
        model.conv1x1_5x5_bias = model_data['conv1x1_5x5_bias']
        model.conv5x5_weights = model_data['conv5x5_weights']
        model.conv5x5_bias = model_data['conv5x5_bias']
        model.conv1x1_pool_weights = model_data['conv1x1_pool_weights']
        model.conv1x1_pool_bias = model_data['conv1x1_pool_bias']
        model.fc_weights = model_data['fc_weights']
        model.fc_bias = model_data['fc_bias']
        
        print(f"Model loaded from {filepath}")
        return model
    
    def predict_single(self, image):

        if len(image.shape) == 2:  # (28, 28)
            image = image.reshape(1, 1, 28, 28)
        elif len(image.shape) == 3:  # (1, 28, 28)
            image = image.reshape(1, 1, 28, 28)
        
        old_print_state = {}
        
        prediction = self.forward(image)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        
        return predicted_digit, confidence, prediction[0]
    
                    
