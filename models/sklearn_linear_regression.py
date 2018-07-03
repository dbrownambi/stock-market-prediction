from sklearn.linear_model import LinearRegression


def linear_regression(df_train, df_test):
    feature_cols = ['CloseA', 'CloseB', 'CloseC', 'CloseD', 'Close']
    X = df_train[feature_cols]
    y = df_train.Next_Day

    # Linear regression using scikit-learn
    lm = LinearRegression()
    lm.fit(X, y)

    # print intercept and coefficients
    print(lm.intercept_)
    print(lm.coef_)

    print(df_test.shape)
    predicted_vals = []
    actual_vals = []
    for i in range(0, len(df_test)):
        expected = lm.coef_[0] * df_test.iloc[i]['CloseA'] + lm.coef_[1] * df_test.iloc[i]['CloseB'] + lm.coef_[2] * \
                                                                                                       df_test.iloc[i][
                                                                                                           'CloseC'] + \
                   lm.coef_[3] * df_test.iloc[i]['CloseD'] + lm.coef_[4] * df_test.iloc[i]['Close'] + lm.intercept_
        print("Predicted: ", expected, "Actual: ", df_test.iloc[i]['Next_Day'])
        predicted_vals.append(expected)
        actual_vals.append(df_test.iloc[i]['Next_Day'])

    return predicted_vals
