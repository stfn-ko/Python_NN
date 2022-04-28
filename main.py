import sys
import numpy as np
import pandas as pd


df = pd.read_csv('data.csv')
X = df.iloc[:,:2].to_numpy() # input-1 & input-2
Y = df.iloc[:,-1:].to_numpy() #answer

X = X/np.amax(X, axis=0) #max of x array
Y = Y*0.5 

Loss = []
EditedOutputArr = np.array([])

class NeuralNetwork(object):
  #par init
  def __init__(self):
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3
    
    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #(3x2 weight matrix from input to hidden layer)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #(3x1 weight matrix from hidden layer to output)


  def feedForward(self, X):
    #forward propogation
    self.z = np.dot(X, self.W1) #dot prod of input(X) and first set of weights(W1)
    self.z2 = self.sigmoid(self.z) #activation func
    self.z3 = np.dot(self.z2, self.W2) #dot prod of hidden layer (z2) and second set of weights(W2)
    output = self.sigmoid(self.z3)
    return output    
    
  
  def backward(self, X, Y, output):
    #back propogation
    self.output_error = Y - output #err in output
    self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

    self.z2_error = self.output_delta.dot(self.W2.T) #z2 err weight on output err
    self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying deriv of sigmoid to z2
    
    self.W1 += X.T.dot(self.z2_delta) #adjust first set of weights(W1)
    self.W2 += self.z2.T.dot(self.output_delta) #adjust second set of weights(W2)
    
    
  def sigmoid(self, s, deriv=False):
    if (deriv == True):
      return s*(1-s)
    return 1/(1 + np.exp(-s))
  
  
  def train(self, X, Y):
    output = self.feedForward(X)
    self.backward(X, Y, output)




NN = NeuralNetwork()

for i in range(10000): #trains NN n-times
  if (i % 100 == 0):
    Loss.append([i, np.mean(np.square(Y - NN.feedForward(X)))])
  NN.train(X, Y)

for f in NN.feedForward(X): #formats output
  if f >= 0.8:
    EditedOutputArr = np.append(EditedOutputArr, 1.0)
  elif f <= 0.1:
    EditedOutputArr = np.append(EditedOutputArr, 0.0)
  else:
    EditedOutputArr = np.append(EditedOutputArr, 0.5)
    

nDF1 = pd.DataFrame(np.c_[X, Y, NN.feedForward(X), EditedOutputArr], 
                   columns=['Input-X', 'Input-Y', 'Expected Output', 
                            'Predicted Output(un-edited)', 'Predicted Output(un-edited)'])

nDF2 = pd.DataFrame(Loss, columns=['Iteration', 'Loss'])
nDF2.set_index("Iteration", inplace = True)

with pd.ExcelWriter("training_outputs\TrIt10-4_HL1_Wt2.xlsx") as writer:
  nDF1.to_excel(writer, sheet_name="Training Data", index=False)
  nDF2.to_excel(writer, sheet_name="Loss Rate", index=True)




