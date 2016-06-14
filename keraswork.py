#This is from the machine learning mastery keras tutorial. I commented the living shit out of this because it's my first attempt at using keras

#testing a new push for github ignore this line
########## imports ##############
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


####### random char generator##########
seed= 7
np.random.seed(seed)


#########loading dataset#############
dataset= np.loadtxt('pima-indians-diabetes.csv', delimiter= ',')
X= dataset[:,0:8] #this prints the first 7 values in the file
#x is input variable
Y= dataset[:,8]
#y is output variable


###### now we will start by making some models. this one will be sequential
#### we add layers to make the network topology better

model= Sequential()

## step 1 is to define input dimensions. This is not an easy thing to do. It is a lot of trial and error. You want it to be big enough to capture the structure at the very least. Here we choose 8. Is input variables?????????
##    Dense = fully connected layers
##    12= number of neurons
##    input dim= see above
##    init= initalization method, want a uniform dist. which is between 0 and 0.05. 'normal' would be  gaussian
###   activation=  activation function. rectifier is best usually for deep learning in bio. Here we use sigmoid at the output layer because we want the output to be between 0 and 1. 

####  I guess we have 3 layers for input, hidden, and output but idk ############

model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

model.add(Dense(8, init='uniform', activation='relu'))

### the model above takes 8 as the number of neurons because that was the output from layer 1 (the input_dim was 8 there)
## model below uses 1 neuron because we want to say 0 or 1 for yes diabetes or no 

model.add(Dense(1, init='uniform', activation='sigmoid'))



### ok we have defined the model. We now compile it.What is that? Read in google docs
### now we will add some extra parameters when we compile to make it better for training 

## loss = binary classification loss standards
## optimizer = its just an efficient default
## metrics = we use this because its classificaion and we want the accuracy reported as the metric

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])



############now its time to execute.....

#we are going to train (fit) the model on our loaded data with model.fit
## epochs = fixed # iterations, need to specify how many - this is pretty small (150)
## batch size = # instances that are evaluated before weight update in network - this is relatively small (10) 

model.fit(X,Y, nb_epoch=150, batch_size=10)



############### now we have to have some kind of verbose output so we know how the model did
## evaluate checks the model
## metrics name will pritn any metrics you decided to configure. here we wanted accuracy


scores= model.evaluate(X,Y)
print ('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
