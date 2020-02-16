import pandas as pd
from watson_machine_learning_client import WatsonMachineLearningAPIClient
import urllib3, requests, json
import requests
import numpy as np

training_data = []
watson_weights = []

def watsonInit():
    global training_data
    global watson_weights
    df = pd.read_csv("../data/MattTrainingData.csv")
    ID = [None for x in range(len(df['Latitude']))]
    lat = df['Latitude']
    long = df['Longitude']
    dist = df['Distance']
    json_array = np.array([ID, lat, long, dist]).T
    # Paste your Watson Machine Learning service apikey here
    # Use the rest of the code sample as written
    apikey = "5O6zEGSxyRXE5rJ4zW4bjRr4QRNpqnb5oK1mXvLRrzy5"

    # Get an IAM token from IBM Cloud
    url     = "https://iam.bluemix.net/oidc/token"
    headers = { "Content-Type" : "application/x-www-form-urlencoded" }
    data    = "apikey=" + apikey + "&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    IBM_cloud_IAM_uid = "bx"
    IBM_cloud_IAM_pwd = "bx"
    response  = requests.post( url, headers=headers, data=data, auth=( IBM_cloud_IAM_uid, IBM_cloud_IAM_pwd ) )
    iam_token = response.json()["access_token"]
    # NOTE: generate iam_token and retrieve ml_instance_id based on provided documentation

    ml_instance_id = 'ed26800a-3b80-45b7-b897-a3053e25cab5'
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + iam_token, 'ML-Instance-ID': ml_instance_id}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line

    payload_scoring = {"input_data": [{"fields": ["ID", "Latitude", "Longitude", "Distance"],
                                   "values": json_array.tolist()}]}

    response_scoring = requests.post(
    'https://us-south.ml.cloud.ibm.com/v4/deployments/0ce79860-7487-4933-adf8-cf976c58d462/predictions',
    json=payload_scoring, headers=header)

    #print("Scoring response")
    watson_response = json.loads(response_scoring.text)

    with open('../data/watsonData.json', 'w') as outfile:
        json.dump(watson_response, outfile)

    dfjson = pd.read_json("../data/watsonData.json")
    hash_tab = dict(dfjson['predictions'])
    array_store = hash_tab[0]['values']



    for i, x in enumerate(array_store):
        if x[0] == 1:
            continue
        else:
            watson_weights.append(x[1][1])
            training_data.append([np.cos(np.radians(lat[i])) * np.cos(np.radians(long[i])) ,
                              np.cos(np.radians(lat[i])) * np.sin(long[i]),
                              np.sin(np.radians(lat[i]))])
    for i, x in enumerate(training_data):
        training_data[i][0] -= np.amin(np.array(training_data)[:,0])
        training_data[i][1] -= np.amin(np.array(training_data)[:,1])
        training_data[i][2] -= np.amin(np.array(training_data)[:,2])
