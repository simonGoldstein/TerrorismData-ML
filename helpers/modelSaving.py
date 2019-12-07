#from tensorflow.keras import Sequential
#from tensorflow.keras.models import load_model
import joblib

#def saveKeras(model, fileName):
#    model.save(fileName)

#def loadKeras(fileName):
#    return load_model(fileName)

def saveSk(model, fileName):
    joblib.dump(model, fileName)

def loadSk(fileName):
    return joblib.load(fileName)
