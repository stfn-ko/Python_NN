import sys
import matplotlib
import numpy as np

inputs = [
            [1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8] 
         ]


weights_1l = [
            [0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5], 
            [-0.26, -0.27, 0.17, 0.87]
          ]
weights_2l = [
            [0.1, -0.14, 0.5], 
            [-0.5, 0.12, -0.33], 
            [-0.44, 0.73, -0.13]
          ]

biases_1l = [2, 3, 0.5]
biases_2l = [-1, 2, -0.5]


layer_1o = np.dot(inputs, np.array(weights_1l).T) + biases_1l
layer_2o = np.dot(layer_1o, np.array(weights_2l).T) + biases_2l
print(layer_2o)
