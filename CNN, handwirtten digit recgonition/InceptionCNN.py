import numpy as np
import pickle

class InceptionCNN:
    def __init__(self, input_channels=1, num_classes=10):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.initialize_weights()
        self.cache = {}
    
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
        batchSize, channels, height, width = inputData.shape
        padded = np.zeros((batchSize, channels, height + 2*padding, width + 2*padding))
        padded[:, :, padding:height+padding, padding:width+padding] = inputData
        return padded
    
    def conv2d_forward(self, inputData, weights, bias, stride=1, padding=0, cache_key=None):
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
        if cache_key:
            self.cache[cache_key] = {
                'input': inputData,
                'weights': weights,
                'bias': bias,
                'output': output,
                'stride': stride,
                'padding': padding
            }
        return output

    def conv2d(self, inputData, weights, bias, stride=1, padding=0):
        return self.conv2d_forward(inputData, weights, bias, stride, padding)

    def relu(self, x):
        return np.maximum(0, x)
    
    def max_pool2d(self, inputData, poolSize=2, stride=2, padding=0):
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

    def inception_block_forward(self, inputData):
        # 1x1 convolution
        conv1x1 = self.conv2d(inputData, self.conv1x1_weights, self.conv1x1_bias)
        conv1x1 = self.relu(conv1x1)
        self.cache['conv1x1'] = conv1x1.copy()

        # 3x3 convolution
        conv3x3 = self.conv2d(inputData, self.conv1x1_3x3_weights, self.conv1x1_3x3_bias)
        conv3x3 = self.relu(conv3x3)
        self.cache['conv1x1_3x3'] = conv1x1_3x3.copy()
        padding = int((3 - 1) / 2)#same padding
        conv3x3 = self.conv2d(conv3x3, self.conv3x3_weights, self.conv3x3_bias, padding=1)
        conv3x3 = self.relu(conv3x3)
        self.cache['conv3x3'] = conv3x3.copy()

        # 5x5 convolution
        conv5x5 = self.conv2d(inputData, self.conv1x1_5x5_weights, self.conv1x1_5x5_bias)
        conv5x5 = self.relu(conv5x5)
        self.cache['conv1x1_5x5'] = conv5x5.copy()
        padding = int((5 - 1) / 2)
        conv5x5 = self.conv2d(conv5x5, self.conv5x5_weights, self.conv5x5_bias, padding=2)
        conv5x5 = self.relu(conv5x5)
        self.cache['conv5x5'] = conv5x5.copy()

        # pooling
        padding = int((3 - 1) / 2)
        pool = self.max_pool2d(inputData, poolSize=3, stride=1, padding=1)
        self.cache['pool'] = pool.copy()
        pool = self.conv2d(pool, self.conv1x1_pool_weights, self.conv1x1_pool_bias)
        pool = self.relu(pool)
        self.cache['conv1x1_pool'] = pool.copy()

        output = np.concatenate((conv1x1, conv3x3, conv5x5, pool), axis=1)
        self.cache['inception_output'] = output.copy()

        return output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, inputData):
        self.cache['input'] = inputData.copy()

        inceptionOutput = self.inception_block_forward(inputData)

        batchSize, channels, height, width = inceptionOutput.shape
        flattened = inceptionOutput.reshape(batchSize, -1)
        self.cache['flattened_shape'] = inceptionOutput.shape
        self.cache['flattened'] = flattened.copy()

        fc_output = np.dot(flattened, self.fc_weights) + self.fc_bias
        self.cache['fc_input'] = flattened.copy()
        self.cache['fc_output'] = fc_output.copy()
        
        #ReLU
        fc_relu = self.relu(fc_output)
        self.cache['fc_relu'] = fc_relu.copy()
        
        #Softmax
        output = self.softmax(fc_relu)
        self.cache['final_output'] = output.copy()
        return output

    def backward(self, y_true, learning_rate=0.001):
        batch_size = y_true.shape[0]
    
        y_pred = self.cache['final_output']
        dL_dSoftmax = y_pred - y_true 
        #softmax cross entropy gradient
        fc_relu = self.cache['fc_relu']
        dL_dFC_relu = dL_dSoftmax.copy()
        dL_dFC_output = dL_dFC_relu.copy()
        dL_dFC_output[fc_relu <= 0] = 0 
        
        #fully connected layer gradients
        fc_input = self.cache['fc_input'] 
        
        dL_dFC_weights = np.dot(fc_input.T, dL_dFC_output) / batch_size
        dL_dFC_bias = np.mean(dL_dFC_output, axis=0)
        
    
        dL_dFC_input = np.dot(dL_dFC_output, self.fc_weights.T)
        inception_shape = self.cache['flattened_shape']
        dL_dInception = dL_dFC_input.reshape(inception_shape)
        
        self.fc_weights -= learning_rate * dL_dFC_weights
        self.fc_bias -= learning_rate * dL_dFC_bias
        dL_conv1x1 = dL_dInception[:, :32, :, :]     
        dL_conv3x3 = dL_dInception[:, 32:64, :, :]
        dL_conv5x5 = dL_dInception[:, 64:96, :, :]   
        dL_pool = dL_dInception[:, 96:128, :, :]    
        
    
        input_data = self.cache['input']
        
        conv1x1_grad = self.simple_conv_weight_gradient(input_data, dL_conv1x1, self.conv1x1_weights.shape)
        conv1x1_bias_grad = np.mean(dL_conv1x1, axis=(0, 2, 3))
        
        self.conv1x1_weights -= learning_rate * conv1x1_grad
        self.conv1x1_bias -= learning_rate * conv1x1_bias_grad

    def simple_conv_weight_gradient(self, input_data, output_grad, weight_shape):
        out_channels, in_channels, kh, kw = weight_shape
        batch_size = input_data.shape[0]
        
        grad = np.zeros(weight_shape)
        
        for b in range(min(batch_size, 8)):
            for c_out in range(out_channels):
                for h in range(min(output_grad.shape[2], 14)): 
                    for w in range(min(output_grad.shape[3], 14)):
                        if h < input_data.shape[2] and w < input_data.shape[3]:
                            input_patch = input_data[b, :, h:h+kh, w:w+kw]
                            if input_patch.shape == (in_channels, kh, kw):
                                grad[c_out] += output_grad[b, c_out, h, w] * input_patch
        
        return grad / batch_size

    def train_step(self, batch_data, batch_labels, learning_rate=0.001):
    
        predictions = self.forward(batch_data)
        
        loss = self.cross_entropy_loss(predictions, batch_labels)
        acc = self.accuracy(predictions, batch_labels)
        
        self.backward(batch_labels, learning_rate)
        return loss, acc

    def cross_entropy_loss(self, predictions, labels):
        batchSize = predictions.shape[0]
        
        epsilon = 1e-12 
        predictions = np.clip(predictions, epsilon, 1. - epsilon)

        loss = -np.sum(labels * np.log(predictions)) / batchSize
        return loss

    def accuracy(self, predictions, labels):
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


