from __future__ import division,print_function

import numpy as np

class Network(object):
    """
    Neural network for softmax regression problems
    """

    def __init__(self,layer_number_of_nodes,layer_activation_functions,layer_has_bias,layer_weight_means_and_stds=None,regularization=None,gamma=0):
        self.layer_number_of_nodes = layer_number_of_nodes           # Of nodes in each layer
        self.layer_activation_functions = [None]
        for act in layer_activation_functions:
            if act=='softmax':
                self.layer_activation_functions.append(self._softmax) 
            if act=='sigmoid':
                self.layer_activation_functions.append(self._sigmoid) 
            if act=='identity':
                self.layer_activation_functions.append(self._identity) 
            if act=='leaky_relu':
                self.layer_activation_functions.append(self._leaky_relu) 
            if act=='gaussian':
                self.layer_activation_functions.append(self._gaussian) 

        self.layer_has_bias = layer_has_bias                         # Whether to add a bias node to each layer
        self.regularization=regularization     
        self.gamma = gamma   

        self.L = len(self.layer_number_of_nodes)                     # Number of layers

        self.weights = [np.array([])]
        self.weight_lengths = [0]

        # Create arrays to hold the weights, which are N_l(+1) by N_(l+1)
        for i in range(self.L-1):
            # if we have a normal distribution and standard deviation, then generate random weights from that distribution,
            if layer_weight_means_and_stds is not None:
                w = layer_weight_means_and_stds[i][1]*np.random.randn(self.layer_number_of_nodes[i] + self.layer_has_bias[i],self.layer_number_of_nodes[i+1]) + layer_weight_means_and_stds[i][0] 
            # Otherwise just initialize the weights to zero
            else:
                w = np.zeros((self.layer_number_of_nodes[i] + self.layer_has_bias[i],self.layer_number_of_nodes[i+1]))
            self.weights.append(w)
            self.weight_lengths.append(len(w.ravel())) 
  
    def feed_forward(self,feature):
        # evaluate the neural network for a vector-valued input
        m = feature.shape[0]

        # Append a column of ones to the input if a bias is desired
        if self.layer_has_bias[0]:
            z = np.column_stack((np.ones((m)),feature))
        else:
            z = feature

        # Initialize lists to hold the node inputs and outputs, treating the input values as the output of the first node
        self.a_vals = [None]
        self.z_vals = [z]

        # Loop over the remaining layers
        for l in range(1,self.L):
            # Take the linear combination of the previous layers outputs (z^(l-1)) and weights (w^(l)) to form a^(l)
            a = np.dot(self.z_vals[l-1],self.weights[l])
            # Run a through the activation function to form z^(l)
            z = self.layer_activation_functions[l](a) 
            # If a bias is desired, append a column of ones to z
            if self.layer_has_bias[l]:         
                z = np.column_stack((np.ones((m)),z))
            # Store these values (for computing the gradient later)
            self.a_vals.append(a) 
            self.z_vals.append(z)
        return z

    def _J_fun(self,feature,label):
        if self.layer_activation_functions[-1]==self._softmax:
            # Model objective function -- Cross-entropy 
             cost_function_data = -np.sum(np.sum(label*np.log(self.feed_forward(feature)),axis=1),axis=0)
        elif self.layer_activation_functions[-1]==self._identity:
            # Model objective function -- SSE 
            cost_function_data = 0.5*np.dot((label-self.feed_forward(feature)).T,(label-self.feed_forward(feature))) 
        else:
            print('Only softmax and identity supported for final layer')            

        # Apply regularization
        if self.regularization=='L1':
            # Don't regularize the bias!
            weight_sums_excluding_bias = [np.sum(abs(w[1:])) for w in self.weights[1:]]
            cost_function_reg = self.gamma*sum(weight_sums_excluding_bias)
        elif self.regularization=='L2':
            weight_sums_excluding_bias = [np.sum((w[1:])**2) for w in self.weights[1:]]
            cost_function_reg = 0.5*self.gamma*sum(weight_sums_excluding_bias)
        else:
            cost_function_reg = 0.0

        return cost_function_data + cost_function_reg

    def _gradient_fun(self,feature,label):
        # Compute the gradient via backpropagation
        m = feature.shape[0]

        # Initialize gradient arrays (same shape as the weights)
        grads = [np.zeros_like(w) for w in self.weights]

        # Compute dJ/da (aka the delta term) for the final layer.  This often involves 
        # Some algebraic simplification when cost function is selected judiciously, so
        # this is coded by hand here.

        l = self.L-1 #last layer

        z = self.z_vals[l]              # Current layer out
        z_previous = self.z_vals[l-1]   # Last layer out
        a = self.a_vals[l]              # Current layer in
        w = self.weights[l]             # Last layer weights
        activation = self.layer_activation_functions[l]     #Current layer activation
        if activation==self._softmax:
            delta_l = (z - label)                       # Current layer error
        elif activation==self._identity:
            delta_l = (z - label)#*activation(a,dx=1)    # Current layer error
        else:
            print('Only softmax and identity supported for final layer') 

        grads[l] = np.dot(z_previous.T,delta_l)    # gradient due to data misfit

        if self.regularization=='L1':
            model_norm = self.gamma*np.sign(w)
            model_norm[0] = 0
        elif self.regularization=='L2':
            model_norm = self.gamma*w
            model_norm[0] = 0
        else:
            model_norm = 0
        grads[l] += model_norm                    # add gradient due to regularization

        # Loop over the remaining layers
        for l in range(self.L-2,0,-1):
        
            z_previous = self.z_vals[l-1]                    # last layer output
            a = self.a_vals[l]                                # Current layer input

            w_next = self.weights[l+1][1:] # weights from the next layer, excluding bias weights
            activation = self.layer_activation_functions[l]  # Current layer activation

            delta_l = np.dot(delta_l,w_next.T)*activation(a,dx=1)  # Current layer error
            grads[l] = np.dot(z_previous.T,delta_l)  # Gradient due to data misfit

            if self.regularization=='L1':
                model_norm = self.gamma*np.sign(self.weights[l])
                model_norm[0] = 0
            elif self.regularization=='L2':
                model_norm = self.gamma*self.weights[l]
                model_norm[0] = 0
            else:
                model_norm = 0
            grads[l] += model_norm             # add gradient due to regularization

        return grads

    # Convert layerwise list of gradients to a vector
    def list_params_to_vector(self,p):
        vals = np.hstack([pp.ravel() for pp in p])
        return vals

    # Convert vector of parameters back to layerwise list
    def vector_params_to_list(self,p):
        vals = [np.array([])]+[p[self.weight_lengths[i]:self.weight_lengths[i]+self.weight_lengths[i+1]].reshape(self.weights[i+1].shape) for i in range(len(self.weight_lengths)-1)]
        return vals
    
    @staticmethod
    def _softmax(X,dx=0):
        if dx==0:
            return np.exp(X)/np.repeat(np.sum(np.exp(X),axis=1,keepdims=True),X.shape[1],axis=1)
    @staticmethod
    def _sigmoid(X,dx=0):
        if dx==0:
            return 1./(1+np.exp(-X))
        if dx==1:
            s = 1./(1+np.exp(-X))
            return s*(1-s)

    @staticmethod
    def _leaky_relu(X,dx=0):
        if dx==0:
            return (X>0)*X + 0.01*(X<=0)*X
        if dx==1:
            return (X>0) + 0.01*(X<=0)

    @staticmethod
    def _identity(X,dx=0):
        if dx==0:
            return X
        if dx==1:
            return np.ones_like(X)

    @staticmethod
    def _gaussian(X,dx=0):
        if dx==0:
            return np.exp(-X**2)
        if dx==1:
            return -2*X*np.exp(-X**2)

            
