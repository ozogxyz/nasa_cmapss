from sklearn.linear_model import LinearRegression


# Basic linear regression
class BaselineRegression(LinearRegression):
    def __init__(self):
        super().__init__()
