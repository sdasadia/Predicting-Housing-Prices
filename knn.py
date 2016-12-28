import numpy as np
from math import sqrt

class k_nearest_neighbours:
    
    print('Following methods are avilable')
    print('1. get_numpy_data(data, features, output)')
    print('2. compute_distances(feature_matrix, feature_vector)')
    print('3. get_k_nearest(k, feature_matrix, feature_vector)')
    print('4. predict_for_one(k, feature_matrix, output_values, feature_vector)')
    print('5. predict_for_set(k, feature_matrix, output_values, feature_set)')
    print('6. get_residual_sum_of_squares(predictions, output)')
    print('7. get_rmse(predictions, output)')
    
    def get_numpy_data(self, data, features, output):
        data['constant'] = 1
        features = ['constant'] + features
        features_array = np.array(data[features])
        output_array = np.array(data[output])
        return(features_array, output_array)
    
    
    def compute_distances(self, feature_matrix, feature_vector):
        diff = feature_matrix[0:] - feature_vector
        distances = np.sqrt(np.sum(diff**2, axis=1))
        return distances
    
    
    def get_k_nearest(self, k, feature_matrix, feature_vector):
        distances = self.compute_distances(feature_matrix, feature_vector)
        index_array = np.argsort(distances)[0:k]
        return index_array
    
    def predict_for_one(self,k, feature_matrix, output_values, feature_vector):
        index_array = self.get_k_nearest(k, feature_matrix, feature_vector)
        predicted_value = np.mean(output_values[index_array])
        return predicted_value
    
    def predict_for_set(self, k, feature_matrix, output_values, feature_set):
        predicted_values = []
        n_houses = feature_set.shape[0]
        for house_index in range(0, n_houses):
            predicted_value = self.predict_for_one(k, feature_matrix, output_values, feature_set[house_index])
            predicted_values.append(predicted_value)
        return predicted_values


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

