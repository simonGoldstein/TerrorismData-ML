# Use this file when running a model file
MODEL_TO_USE = "final_project_knn"

import importlib.util

def loadModel(modelName = MODEL_TO_USE):
    spec = importlib.util.spec_from_file_location(MODEL_TO_USE, "models/" + MODEL_TO_USE + ".py")
    tempModule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tempModule)

if __name__ == "__main__":
    loadModel(MODEL_TO_USE)