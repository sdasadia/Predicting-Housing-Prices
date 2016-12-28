import numpy as np
from math import sqrt

class lasso_regression:
    
    print('Following methods are avilable')
    print('1. get_numpy_data(data, features, output)')
    print('2. predict_output(feature_matrix, weights)')
    print('3. normalize_features(feature_matrix)')
    print('4. lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)')
    print('5. lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance)')
    print('6. get_residual_sum_of_squares(predictions, output)')
    print('7. get_rmse(predictions, output)')
    
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
    
    
    def normalize_features(self,feature_matrix):
        norms = np.linalg.norm(feature_matrix, axis=0)
        normalized_features = feature_matrix / norms
        return(normalized_features, norms)


    def lasso_coordinate_descent_step(self, i, feature_matrix, output, weights, l1_penalty):
        
        # compute prediction
        prediction = predict_output(feature_matrix, weights)
        
        # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
        ro_i = np.dot(feature_matrix[:, i], (output - prediction + weights[i] * feature_matrix[:, i]))
    
        if i == 0: # intercept -- do not regularize
            new_weight_i = ro_i
        elif ro_i < -l1_penalty/2.:
            new_weight_i = ro_i + l1_penalty/2.
        elif ro_i > l1_penalty/2.:
            new_weight_i = ro_i - l1_penalty/2.
        else:
            new_weight_i = 0.
    
        return new_weight_i

    def lasso_cyclical_coordinate_descent(self, feature_matrix, output, initial_weights, l1_penalty, tolerance):
        converged = False
        weights = np.array(initial_weights) # make sure it's a numpy array
        while not converged:
            converged = True
            for i in range(len(weights)):
                old_weights_i = weights[i] # old value of weight[i], as it will be overwritten
                # the following line uses new values for weight[0], weight[1], ..., weight[i-1]
                # and old values for weight[i], ..., weight[d-1]

weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            
            
                # use old_weights_i to compute change in coordinate
                change_i = abs(weights[i] - old_weights_i)
                if change_i >= tolerance:
                    converged = converged & False
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

