import numpy as np

class linear_regression:
    
    print('Following methods are avilable')
    print('1. linear_regression_fit(input_feature, output')
    print('2. get_regression_predictions(input_feature intercept, slope)')
    print('3. get_resodual_sum_of_squares(self, input_feature, output, intercept, slope)')
    
    def linear_regression_fit(self,input_feature, output):

        a = np.array(input_feature)
        b = np.array(output)
        N = a.shape[0]
        
        input_output_prod = a * b
        input_squared = a * a
        
        # Calculate slope and intercept
        slope = (input_output_prod.sum() - (a.sum() * b.sum())/N) / (input_squared.sum() - (a.sum() * a.sum())/N)    
        intercept = b.mean() - slope * a.mean()
        
        return (intercept, slope)
    
    
    def get_regression_predictions(self, input_feature, intercept, slope):
    
        a = np.array(input_feature)    
        predicted_values = intercept + slope * a
    
        return predicted_values
    
    def get_residual_sum_of_squares(self, input_feature, output, intercept, slope):
    
        a = np.array(input_feature)
        b = np.array(output)
    
        # First get the predictions
        predicted_values = intercept + slope * a

        # Compute the residuals
        residuals = b - predicted_values

        # square the residuals and add them up
        residuals_squared = residuals * residuals
    
        RSS = residuals_squared.sum()

        return(RSS)
