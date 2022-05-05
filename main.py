import numpy as np
import pandas as pd


df = pd.read_csv('data.csv')
input = df.iloc[:,:2].to_numpy() # input-1 & input-2
answer = df.iloc[:,-1:].to_numpy() #answer

input = input/np.amax(input, axis=0) #max of x array
answer = answer*0.5 

Loss = []
EditedOutputArr = np.array([])

class NeuralNetwork(object):
  #par init
  def __init__(self):
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 5
    
    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #(2x3) weight matrix from input to hidden layer
    self.b1 = np.zeros(self.hiddenSize) # 1x3 bias for 1st set of weights
    
    self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize) #(3x3) weight matrix from hidden layer one to hidden layer two
    self.b2 = np.zeros(self.hiddenSize) # 1x3 bias for 2nd set of weights
    
    self.W3 = np.random.randn(self.hiddenSize, self.outputSize) #(3x1) weight matrix from hidden layer to output
    self.b3 = np.zeros(self.outputSize) # 1x1 bias for 3rd set of weights
    

  def feedForward(self, input):
    #forward propogation
    self.z1 = np.dot(input, self.W1) + self.b1 #dot prod of input(X) and first set of weights(W1)
    self.a1 = self.sigmoid(self.z1) #activation func
    
    self.z2 = np.dot(self.a1, self.W2) + self.b2 #dot prod of the first hidden layer and second set of weights(W2)
    self.a2 = self.sigmoid(self.z2) #activation func
    
    self.z3 = np.dot(self.a2, self.W3) + self.b3 #dot prod of the second hidden layer and third set of weights(W3)
    output = self.sigmoid(self.z3)
    
    return output    
    
  
  def backward(self, input, answer, output):
    #back propogation
    self.output_error = answer - output #err in output
    self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

    self.a2_error = self.output_delta.dot(self.W3.T) #a2 err weight on output err
    self.a2_delta = self.a2_error * self.sigmoid(self.a2, deriv=True) #applying deriv of sigmoid to a2

    self.a1_error = self.a2_delta.dot(self.W2.T) #a err weight on a2 err
    self.a1_delta = self.a1_error * self.sigmoid(self.a1, deriv=True) #applying deriv of sigmoid to a
    
    self.W1 += input.T.dot(self.a1_delta) #adjust first set of weights(W1)
    self.W2 += self.a1.T.dot(self.a2_delta) #adjust second set of weights(W2)
    self.W3 += self.a2.T.dot(self.output_delta) #adjust third set of weights(W3)
    
    
  def sigmoid(self, s, deriv=False):
    if (deriv == True):
      return s*(1-s)
    return 1/(1 + np.exp(-s))
  
  
  def train(self, input, answer):
    output = self.feedForward(input)
    self.backward(input, answer, output)




NN = NeuralNetwork()

for i in range(1000): #trains NN n-times
  if (i % 10 == 0):
    Loss.append([i, np.mean(np.square(answer - NN.feedForward(input)))])
  NN.train(input, answer)



#Documentation

for f in NN.feedForward(input): #interprets output
  if f >= 0.8:
    EditedOutputArr = np.append(EditedOutputArr, 1.0)
  elif f <= 0.1:
    EditedOutputArr = np.append(EditedOutputArr, 0.0)
  else:
    EditedOutputArr = np.append(EditedOutputArr, 0.5)
    

nDF1 = pd.DataFrame(np.c_[input, answer, NN.feedForward(input), EditedOutputArr], 
                   columns=['Input-X', 'Input-Y', 'Expected Output', 
                            'Predicted Output(un-edited)', 'Predicted Output(un-edited)'])

nDF2 = pd.DataFrame(Loss, columns=['Iteration', 'Loss'])
nDF2.set_index("Iteration", inplace = True)

with pd.ExcelWriter("training_outputs\TrIt_-_.xlsx") as writer:
  nDF1.to_excel(writer, sheet_name="Training Data", index=False)
  nDF2.to_excel(writer, sheet_name="Loss Rate", index=True)




