import pickle
import json
import numpy
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model
from surprise import Dataset, evaluate
from surprise import KNNBasic
import os
import urllib.request
import azurepipelines_optimizely_sdk as aps
import logging
import sys

def get_data(model):
    # manually downloading the file, as it requires a prompt otherwise
    url='http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    DATASETS_DIR = os.path.expanduser('~') + '/.surprise_data/'

    logger.info("Starting processing model")

    name = 'ml-100k'
    os.makedirs(DATASETS_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, DATASETS_DIR + 'tmp.zip')


    import zipfile
    with zipfile.ZipFile(DATASETS_DIR + 'tmp.zip', 'r') as tmp_zip:
        tmp_zip.extractall(DATASETS_DIR + name)

    data = Dataset.load_builtin(name)
    trainingSet = data.build_full_trainset()
    testSet = trainingSet.build_anti_testset()
    predictions = model.test(testSet)

    return predictions


from collections import defaultdict
 
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

import os, io
 
def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
 
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
 
    return rid_to_name

def initLogger():
    global logger
    logging.basicConfig(filename="newfile.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w')
    #Creating an object 
    logger=logging.getLogger() 
  
    #Setting the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG)  

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def init():
    global modelRecommendationByName
    global rid_to_name
    global azurePipelineOptimizelySdk

    initLogger()

    PROJECT_ID = "12098094739"
    EXPERIMENT_KEY = "Model_Experiment"
    azurePipelineOptimizelySdk = aps.AzurePipelinesOptimizelySdk(PROJECT_ID, EXPERIMENT_KEY)

    modelFileByName = {
        "modelA" : "model.pkl"
    }

    modelRecommendationByName = {}
    for modelName, modelFileName in modelFileByName.items():
        logger.info("Predicting corresponding to model : " + modelName)
        model_path = Model.get_model_path(modelFileName)
        model = joblib.load(model_path)
        predictions = get_data(model)
        top3_recommendations = get_top3_recommendations(predictions)
        modelRecommendationByName[modelName] = top3_recommendations
    
    rid_to_name = read_item_names()

def run(raw_data):

    jsonData = json.loads(raw_data)
    userUid = jsonData['uid']

    # Integegration with optimizely
    variationKey = azurePipelineOptimizelySdk.getVariationKey(userUid)
    if variationKey is None:
        variationKey = list(modelFileByName.keys())[0]
    print(variationKey)
    logger.info("Predicting for user '" + userUid + "' using model : " + variationKey)
    top3_recommendations = modelRecommendationByName[variationKey]

    #data = numpy.array(data)
    for uid, user_ratings in top3_recommendations.items():
        try:
            if str(uid) == str(userUid):
                result = str((uid, [rid_to_name[iid] for (iid, _) in user_ratings]))
        except Exception as e:
            result = str(e)
    return json.dumps({"result": result})
