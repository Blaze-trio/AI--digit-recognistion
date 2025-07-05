import numpy as np

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
        # 1X1 convolution layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 32, 1, 1)
        self.conv1x1_weights = self.xavier_uniform(fan_in,fan_out, (32,self.input_channels, 1, 1))
        self.conv1x1_bias = np.zeros((32,))

        # 3X3 convolution layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels,32, 1, 1)
        self.conv1x1_3x3_weights = self.xavier_uniform(fan_in, fan_out, (32, self.input_channels, 1, 1))
        self.conv1x1_3x3_bias = np.zeros((32,))
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 32, 3, 3)
        self.conv3x3_weights = self.xavier_uniform(fan_in, fan_out, (32, self.input_channels, 3, 3))
        self.conv3x3_bias = np.zeros((32,))

        # 5X5 convolution layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 32, 1, 1)
        self.conv1x1_5x5_weights = self.xavier_uniform(fan_in, fan_out, (32, self.input_channels, 1, 1))
        self.conv1x1_5x5_bias = np.zeros((32,))
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 32, 5, 5)
        self.conv5x5_weights = self.xavier_uniform(fan_in, fan_out, (32, self.input_channels, 5, 5))
        self.conv5x5_bias = np.zeros((32,))

        # max pooling layer
        fan_in, fan_out = self.calculate_conv_fan_in_out(self.input_channels, 32, 3, 3)
        self.conv1x1_pool_weights = self.xavier_uniform(fan_in, fan_out, (32, self.input_channels, 1, 1))
        self.conv1x1_pool_bias = np.zeros((32,))

        # fully connected layer
        fc_input_size = 128 * 28 * 28 
        fc_output_size = self.num_classes

        fan_in, fan_out = fc_input_size, fc_output_size
        self.fc_weights = self.xavier_uniform(fan_in, fan_out, (fc_output_size, fc_input_size))
        self.fc_bias = np.zeros((fc_output_size,))
    
    def pad_input(self, inputData, padding):
        batch_size, channels, height, width = inputData.shape
        padded = np.zeros((batch_size, channels, height + 2*padding, width + 2*padding))
        padded[:, :, padding:height+padding, padding:width+padding] = input_data
        return padded
    
    def conv2d(self, inputData, weights, bias, stride=1, padding=0):
        if padding > 0:
            inputData = self.pad_input(inputData, padding)

        batch_size, inChannels, inHeight, inWidth = inputData.shape
        outChannels, _, filterHeight, filterWidth = weights.shape

        outHeight = int((inHeight - filterHeight) / stride) + 1
        outWidth = int((inWidth - filterWidth) / stride) + 1

        output = np.zeros((batch_size, outChannels, outHeight, outWidth))

        for b in range(batch_size):
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
        if padding > 0:
            inputData = self.pad_input(inputData, padding)

        batch_size, channels, inHeight, inWidth = inputData.shape
        outHeight = int((inHeight - poolSize) / stride) + 1
        outWidth = int((inWidth - poolSize) / stride) + 1

        output = np.zeros((batch_size, channels, outHeight, outWidth))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(outHeight):
                    for w in range(outWidth):
                        hStart = h * stride
                        hEnd = hStart + poolSize
                        wStart = w * stride
                        wEnd = wStart + poolSize
                        output[b, c, h, w] = np.max(inputData[b, c, hStart:hEnd, wStart:wEnd])
        return output
