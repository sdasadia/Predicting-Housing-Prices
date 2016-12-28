import numpy as np
from math import sqrt

class ridge_regression:
    
    print('Following methods are avilable')
    print('1. get_numpy_data(data, features, output)')
    print('2. predict_output(feature_matrix, weights)')
    print('3. feature_derivative(errors, feature)')
    print('4. regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)')
    print('5. get_residual_sum_of_squares(predictions, output)')
    print('6. get_rmse(predictions, output)')
    
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
    
    
    def feature_derivative_ridge(self,errors, feature, weight, l2_penalty, feature_is_constant):
        # If feature_is_constant is True, derivative is twice the dot product of errors and feature
        # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    
        if feature_is_constant == True:
            derivative = 2 * np.dot(errors, feature)
        else:
            derivative = 2 * np.dot(errors, feature) + 2*l2_penalty*weight
        return derivative
            

    def ridge_regression_gradient_descent(self, feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    
        weights = np.array(initial_weights)
        iterations = 1
    
        while iterations < max_iterations:
        
            predictions = self.predict_output(feature_matrix, weights)
            errors = predictions - output
        
            for i in xrange(len(weights)): # loop over each weight
                if i == 0:
                    derivative = self.feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, True)
                else:
                    derivative = self.feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, False)
            
                weights[i] = weights[i] - step_size * derivative
        
            iterations = iterations + 1
        return weights

            
    def get_residual_sum_of_squares(self, predictions, output):
    
        residual = output - predictions
        residual_squared = residual * residual
        RSS = residual_squared.sum()
        return(RSS)

    def get_rmse(self, predictions, output):
    
        error = predictions - output
        val_err = np.dot(error,error)
        val_err = np.sqrt(val_err/len(output))
        return val_err

