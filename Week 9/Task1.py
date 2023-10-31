from sklearn.model_selection import train_test_split
import os

class Task1:
    def __init__(self, dir):
        self.dir = dir
        self.files = {}
        self.total = 0

        self.X_val= []
        self.X_test = []
        self.y_val = []
        self.y_test = []
        
        # Reading the labels and storing them in a dictionary with the key as the category and the value as the file name
        for file in os.listdir(dir):
            with open(dir + '/' + file, 'r') as f:
                labels = f.readlines()
                self.total += len(labels)
                for label in labels:
                    catergory = int(label.split(' ')[0])
                    if catergory not in self.files:
                        self.files[catergory] = []
                    self.files[catergory].append(file[:-3]+'jpg')

        # Splitting the data into validation and test sets with 50% of the data in each
        keys = list(self.files.keys())
        
        for key in keys:
            X_val, X_test, y_val, y_test = train_test_split(self.files[key], [key]*len(self.files[key]), test_size=0.5, random_state=0, shuffle=True)

            self.X_val += X_val
            self.X_test += X_test
            self.y_val += y_val
            self.y_test += y_test

    def getSplits(self):
        return self.X_val, self.X_test, self.y_val, self.y_test
    
    def __len__(self):
        return self.total
    
    


if __name__ == '__main__':
    dir = './valid/labels'
    task1 = Task1(dir)
    X_val,X_test,y_val,y_test = task1.getSplits()
    print(task1.__len__())




