import numpy as np
from math import sqrt

class multiple_regression:
    
    print('Following methods are avilable')
    print('1. get_numpy_data(data, features, output)')
    print('2. predict_output(feature_matrix, weights)')
    print('3. feature_derivative(errors, feature)')
    print('4. regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)')
    print('5. get_rmse(predictions, output)')
    
    def get_numpy_data(self, data, features, output):
    
        # Add constant column to the data
        data['constant'] = 1
        features = ['constant'] + features
    
        features_array = np.array(data[features])
        output_array = np.array(data[output])
        
        return(features_array, output_array)
    
    
    def predict_output(self, feature_matrix, weights):
    
        # predictions vector is the dot product of features and weights
        predictions = np.dot(feature_matrix, weights)
    
        return(predictions)
    
    def feature_derivative(self, errors, feature):
    
        # derivative =  twice the dot product of error and features matrix
        derivative = 2*np.dot(errors,feature)
        
        return(derivative)

    def regression_gradient_descent(self, feature_matrix, output, initial_weights, step_size, tolerance):
        converged = False
        weights = np.array(initial_weights) # converts initial_weights to a numpy array
    
        while not converged:
        
            # 1. Computer the predictions  and errors using initial_weights
            predictions = self.predict_output(feature_matrix, weights)
            errors = predictions - output
        
            gradient_sum_squares = 0 # initialize the gradient sum of squares
        
            for i in range(len(weights)): # loop over each weight
            
                # compute the derivative for weight[i]:
                derivative = self.feature_derivative(errors, feature_matrix[:, i])
            
                # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
                gradient_sum_squares = gradient_sum_squares + (derivative * derivative)
            
                # subtract the step size times the derivative from the current weight
                weights[i] = weights[i] - step_size * derivative
        
            # compute the square-root of the gradient sum of squares to get the gradient matnigude:
            gradient_magnitude = sqrt(gradient_sum_squares)
            if gradient_magnitude < tolerance:
                converged = True
    
        return(weights)


    def get_rmse(self, predictions, output):
    
        error = predictions - output
        val_err = np.dot(error,error)
        val_err = np.sqrt(val_err/len(output))
        return val_err

