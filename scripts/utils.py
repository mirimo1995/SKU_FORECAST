from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def modelresults(target, predictions):
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    
    print('Mean absolute error on model is {:.4f}'.format(mae))
    print('')
    print('The r2 score on model is {:.4f}'.format(r2))
    
    return mae, r2