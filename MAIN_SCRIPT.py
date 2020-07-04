from models import K_SVM, KernelLogReg, KernelRidgeReg
from utils import get_saved_data, save_predictions, reformat_data

if __name__ == '__main__':
    # Kernels considered
    methods = ['WD_d10', 'SP_k6']

    # Import data
    data, kernels, ID = get_saved_data(methods)

    print('\n\n')
    print('Getting the data.........')
    X_train, y_train, X_val, y_val, X_test, kernels, ID = reformat_data(data, kernels, ID)

    print('Fitting the model.....')
    model = KernelLogReg(kernels[0], ID, lambda_=0.1, solver='BFGS')
    model.fit(X_train, y_train)

    print('Making predictions .......')
    print('\n\nAccuracy on validation set 1: {:0.4f}'.format(model.score(model.predict(X_val), y_val)))

    # Compute predictions
    y_pred = save_predictions(model, X_test)

    print('\n\nPredictions ok')
    print('------')
    print('Check prediction file Yte.csv in folder')
