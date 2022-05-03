import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')
input = df.iloc[:, :2].to_numpy()  # input-1 & input-2
answer = df.iloc[:, -1:].to_numpy()  # answer
Loss = []


class NN(object):
    # constructor
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.h1_layerSize = 5  # amount of neurons in 1st hidden layer
        self.h2_layerSize = 3  # amount of neurons in 2nd hidden layer

    #weights and biases
        # (2x5) weights for input to hidden1 layer
        self.W1 = np.random.randn(self.inputSize, self.h1_layerSize)
        self.b1 = np.zeros(self.h1_layerSize)  # 1x5 bias for the 1st layer

        # (5x3) weights for input to hidden1 layer
        self.W2 = np.random.randn(self.h1_layerSize, self.h2_layerSize)
        self.b2 = np.zeros(self.h2_layerSize)  # 1x3 bias for the 2nd layer

        # (3x1) weights for input to hidden1 layer
        self.W3 = np.random.randn(self.h2_layerSize, self.outputSize)
        self.b3 = np.zeros(self.outputSize)  # 1x1 bias for the 3rd layer

 #

    def forwardP(self, input):
        self.z1 = np.dot(input, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.z1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.z2, self.W3) + self.b3
        output = self.sigmoid(self.z3)

        return output

#

    def backwardP(self, input, answer, output):
        # back propogation
        self.output_error = answer - output  # err in output
        self.output_delta = self.output_error * \
            self.sigmoid(output, deriv=True)

        self.a2_error = self.output_delta.dot(
            self.W3.T)  # a2 err weight on output err
        # applying deriv of sigmoid to a2
        self.a2_delta = self.a2_error * self.sigmoid(self.a2, deriv=True)

        self.a1_error = self.a2_delta.dot(self.W2.T)  # a err weight on a2 err
        # applying deriv of sigmoid to a
        self.a1_delta = self.a1_error * self.sigmoid(self.a1, deriv=True)

        # adjust first set of weights(W1)
        self.W1 += input.T.dot(self.a1_delta)
        # adjust second set of weights(W2)
        self.W2 += self.a1.T.dot(self.a2_delta)
        # adjust third set of weights(W3)
        self.W3 += self.a2.T.dot(self.output_delta)


#


    def sigmoid(self, s, deriv=False):
        if deriv == True:
            return s*(1-s)
        return 1/(1 + np.exp(-s))


#

    def train(self, input, answer):
        output = self.forwardP(input)
        self.backwardP(input, answer, output)


#
NN = NN()

for i in range(1000):  # trains NN n-times
    if (i % 10 == 0):
        Loss.append([i, np.mean(np.square(answer - NN.forwardP(input)))])
    NN.train(input, answer)

nDF1 = pd.DataFrame(np.c_[input, answer, NN.forwardP(input)],
                    columns=['Input-X', 'Input-Y', 'Expected Output',
                             'Predicted Output'])
nDF2 = pd.DataFrame(Loss, columns=['Iteration', 'Loss'])
nDF2.set_index("Iteration", inplace=True)

with pd.ExcelWriter("training_outputs\foo.xlsx") as writer:
    nDF1.to_excel(writer, sheet_name="Training Data", index=False)
    nDF2.to_excel(writer, sheet_name="Loss Rate", index=True)
