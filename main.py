import numpy as np
import pandas as pd

# Data frame init & input/output normalization
df = pd.read_csv('data.csv')
input = df.iloc[:,:2].to_numpy() # input-1 & input-2
input = input/np.amax(input, axis=0) #max of x array
answer = df.iloc[:,-1:].to_numpy() #answer
answer = answer*0.5 
Loss = []

#
class NeuralNetwork(object):
  #parameter initilization
  def __init__(self):
    self.inputSize = 2
    self.outputSize = 1
    #amount of weights in hidden layer
    self.hiddenSize = 5 
    
    #weights & biases
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #(2x3) weight matrix from input to hidden layer 1
    self.b1 = np.zeros(self.hiddenSize) # 1x3 bias for 1st set of weights
    
    self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize) #(3x3) weight matrix from hidden layer 1 to hidden layer 2
    self.b2 = np.zeros(self.hiddenSize) # 1x3 bias for 2nd set of weights
    
    self.W3 = np.random.randn(self.hiddenSize, self.outputSize) #(3x1) weight matrix from hidden layer 2 to output
    self.b3 = np.zeros(self.outputSize) # 1x1 bias for 3rd set of weights
    
  #
  def feedForward(self, input):
    #forward propogation
    self.z1 = np.dot(input, self.W1) + self.b1 #dot prod of input(X) and first set of weights(W1)
    self.a1 = self.sigmoid(self.z1) #activation func
    
    self.z2 = np.dot(self.a1, self.W2) + self.b2 #dot prod of hidden layer 1 and second set of weights(W2)
    self.a2 = self.sigmoid(self.z2) #activation func
    
    self.z3 = np.dot(self.a2, self.W3) + self.b3 #dot prod of hidden layer 2 and third set of weights(W3)
    output = self.sigmoid(self.z3)
    
    return output    
    
  #
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
    
  # 
  def sigmoid(self, s, deriv=False):
    if (deriv == True):
      #sigmoid's derived function
      return s*(1-s) 
    return 1/(1 + np.exp(-s))
  
  #
  def train(self, input, answer):
    output = self.feedForward(input)
    self.backward(input, answer, output)


NN = NeuralNetwork()

for i in range(1000): #trains NN n-epochs
  if (i % 10 == 0):
    #loss rate (100 samples)
    Loss.append([i, np.mean(np.square(answer - NN.feedForward(input)))])
  NN.train(input, answer)


#Output documentation into tables
#Spreadsheet for i/o information and training results
nDF1 = pd.DataFrame(np.c_[input, answer, NN.feedForward(input)], 
                   columns=['Input-X', 'Input-Y', 'Expected Output', 
                            'Predicted Output'])
#Spreadsheet for loss rate
nDF2 = pd.DataFrame(Loss, columns=['Iteration', 'Loss'])
nDF2.set_index("Iteration", inplace = True)

# Puts the whole thing into one .xlsx file
# notation for __config__ : H*_HL*_$, where 
# H* = amount of neurons per hidden layer, e.g.: H3
# HL* = amount of hidden layers in NN, e.g.: HL2
# $ = type of activtion function used, e.g.: SIG (sigmoid)
#
# notation for __epoch__, where TrIt@-^, where
# @ = base of scientific notation (>10)
# ^ = exponent of scientific notation (integer), e.g.: TrIt10-7

with pd.ExcelWriter("training_outputs\__config__\__epoch__.xlsx") as writer:
  nDF1.to_excel(writer, sheet_name="Training Data", index=False)
  nDF2.to_excel(writer, sheet_name="Loss Rate", index=True)




