from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


class LogisticModel():
    def __init__(self, train_X, train_y):
        self.type = self.__class__.__name__
        self.X = train_X
        self.y = train_y
        self.clf = LogisticRegression(random_state=0).fit(self.X, self.y)

    def predict(self, pred_x):
        pred_y = self.clf.predict_proba(pred_x)
        return pred_y


class LinearModel():
    def __init__(self, train_X, train_y):
        self.type = self.__class__.__name__
        self.X = train_X
        self.y = train_y
        self.clf = LinearRegression().fit(self.X, self.y)

    def predict(self, pred_x):
        pred_y = self.clf.predict(pred_x)
        return pred_y
