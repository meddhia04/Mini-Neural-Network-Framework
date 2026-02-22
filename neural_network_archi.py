import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size,learning_rate = 0.01):
        '''
        input_size: number of features
        hidden_size: number of neurons in middle layer
        output_size: Number of classes to predict 
        learning_rate: how big is the steps during training
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size  = output_size
        
        #set random seed for reproducibility to ensure we get the same random number every program run
        np.random.seed(42)
        '''
        2)layer one: input to hidden
        weight and bias initialization first
        '''
        
        #multiplying by 0.01 to prevent saturation
        self.W1 = np.random.randn(input_size,hidden_size)*0.01
        
        #b1 : initialize bias one bias per hidden neuron
        self.b1 = np.zeros((1,hidden_size))
        
        """
        3)layer two: hidden to output layer
        """
        self.W2 = np.random.randn(hidden_size,output_size)*0.01
        #bias per output class
        self.b2 = np.zeros((1,output_size))
        #here we store losses during training it will be useful for plotting the curve
        self.losses = []
        
        