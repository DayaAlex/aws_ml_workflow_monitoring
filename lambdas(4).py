import boto3
import base64

s3 = boto3.client('s3')
BUCKET_NAME = 'sagemaker-studio-963854469832-csyjvdk511'
PREFIX = 'first_lambda'

#image_serialisation_lambda_function
def lambda_handler(event, context):
    key = event['body']['s3_key']#getting address from step function event input
    bucket = BUCKET_NAME
    
    file_name = '/tmp/image.png'
    s3.download_file(bucket, key, file_name)

    with open('/tmp/image.png', 'rb') as f:
        image_data = base64.b64encode(f.read())
    
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body' : 
        {
            'image_data': image_data,
            's3_bucket': bucket,
            's3_key': key,
            'inferences': []
        }
    } 

import sagemaker
import base64
import json
from sagemaker.serializers import IdentitySerializer

ENDPOINT = 'image-classification-2024-08-25-01-44-20-201'

# sagemaker_inference_lambda function
def lambda_handler(event, context):
    image = base64.b64decode(event['body']['image_data'])
    endpoint = ENDPOINT
    predictor = sagemaker.predictor.Predictor(
        endpoint,
        sagemaker_session=sagemaker.Session(),
    )
    predictor.serializer = IdentitySerializer("image/png")

    inferences = predictor.predict(image)
    event['body']['inferences'] = inferences.decode('utf-8')  # classification task yields text output

    return {
        'statusCode': 200,
        'body': json.dumps(event['body'])
    }

import json
THRESHOLD = 0.93
#filter_out_low_confidence_inferences
def lambda_handler(event, context):
    meets_threshold = None
    body = json.loads(event['body']) #the b strings are stored as json, so need to be convereted to python lists/dictionaries
    inferences = json.loads(body['inferences'])

    for inference in inferences:
        if inference > THRESHOLD:
            meets_threshold = True
    if meets_threshold:
        pass
    else:
        raise Exception('THRESHOLD_CONFIDENCE_NOT_MET')
    
    return{
        'statusCode': 200,
        'body': json.dumps(event)
    }
