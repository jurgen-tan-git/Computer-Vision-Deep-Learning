from sklearn.model_selection import train_test_split
import os

class Task1:
    def __init__(self, dir):
        self.dir = dir
        self.files = os.listdir(dir)
        self.names = {}
        self.total = 0

        self.X_val= []
        self.X_test = []
        self.y_val = []
        self.y_test = []
        
        # Reading the labels and storing them in a dictionary with the key as the category and the value as the file name
        for file in self.files:
            category = file.split('-')[:3]
            if category[2].isnumeric() == True:g
                category = category[:2]
            category = '-'.join(category)
            if category not in self.names:
                self.names[category] = []
            self.names[category].append(file)
    
        # Splitting the data into validation and test sets with 50% of the data in each
        for key in self.names.keys():
            X_val, X_test, y_val, y_test = train_test_split(self.names[key], [key]*len(self.names[key]), test_size=0.5, random_state=0, shuffle=True)
            # print(len(X_val), len(X_test), len(y_val), len(y_test))
            self.X_val += X_val
            self.X_test += X_test
            self.y_val += y_val
            self.y_test += y_test

    def getSplits(self):
        return self.X_val, self.X_test, self.y_val, self.y_test
    
    def __len__(self):
        return self.names.__len__()
    
    


if __name__ == '__main__':
    dir = './valid/images'
    task1 = Task1(dir)
    X_val, X_test, y_val, y_test = task1.getSplits()
    
    print(len(X_val), len(X_test), len(y_val), len(y_test))
    print(task1.__len__())






