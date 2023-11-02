from Task1 import *
import shutil

if __name__ == '__main__':
    dir = './valid/images'
    task1 = Task1(dir)
    X_val, X_test, y_val, y_test = task1.getSplits()
    
    # Copy the file from the source to the destination
    try:
        print('Creating directories')
        os.mkdir('./traffic')
        
        os.mkdir('./traffic/val')
        os.mkdir('./traffic/val/images')
        os.mkdir('./traffic/val/labels')
        
        os.mkdir('./traffic/test')
        os.mkdir('./traffic/test/images')
        os.mkdir('./traffic/test/labels')
 
    except:
        pass

    
    for file in X_val:
        
        shutil.copy('./valid/images/'+file, './traffic/val/images/'+file)
        shutil.copy('./valid/labels/'+file[:-3]+'txt', './traffic/val/labels/'+file[:-3]+'txt')
    for file in X_test:
        shutil.copy('./valid/images/'+file, './traffic/test/images/'+file)
        shutil.copy('./valid/labels/'+file[:-3]+'txt', './traffic/test/labels/'+file[:-3]+'txt')
