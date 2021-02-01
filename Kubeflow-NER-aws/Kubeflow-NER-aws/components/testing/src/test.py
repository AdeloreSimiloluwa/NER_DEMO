import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import argparse
import os
import json
import plac
import random
import warnings
from pathlib import Path
import spacy
import boto3
import botocore
import logging
import tempfile


argv=None
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--model_dir')
parser.add_argument('--output-model-path-file', help='')
known_args, pipeline_args = parser.parse_known_args(argv)

#upload data to s3
def upload_folder_to_s3(s3bucket, inputDir, s3Path):
            print("Uploading results to s3 initiated...")
            print("Local Source:",inputDir)
            os.system("ls -ltR " + inputDir)
            print("Dest  S3path:",s3Path)
            try:
                for path, _, files in os.walk(inputDir):
                    for file in files:
                        dest_path = path.replace(inputDir,"")
                        __s3file = os.path.normpath(s3Path + '/' + dest_path + '/' + file)
                        __local_file = os.path.join(path, file)
                        print("upload : ", __local_file, " to Target: ", __s3file, end="")
                        s3bucket.upload_file(__local_file, __s3file)
                        print(" ...Success")
            except Exception as e:
                print(" ... Failed!! Quitting Upload!!")
                print(e)
                raise e

s3 = boto3.resource('s3', aws_access_key_id='', 
                                    aws_secret_access_key='')
s3bucket = s3.Bucket('spacy-ner-test-bucket')

# download training data from s3
def download_s3_file(bucket_name, s3_file):
    try:
        s3_client = boto3.client('s3', aws_access_key_id='', 
                                 aws_secret_access_key='')
        downloaded_file = f'/data/{s3_file}'
        import pathlib
        full_path_with_dir = pathlib.Path(downloaded_file)
        if not os.path.exists(full_path_with_dir.parent):
            os.makedirs(full_path_with_dir.parent)
        logging.info(f'parent path: {full_path_with_dir.parent}')
        logging.info(f'downloaded_file: {downloaded_file}')
        logging.info(f'bucket: {bucket_name}')
        logging.info(f'bucket: {s3_file}')
        s3_client.download_file(bucket_name, s3_file, downloaded_file)
        return downloaded_file
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logging.fatal(f"Object does not exist")
        else:
            raise
    print('downloading')

#download model folder from s3
def download_s3_folder(bucket_name, s3_folder):
    # ++++++++++++++++++++++++++++++++++++++++++++++
    # Download bucket directory content from S3
    # ++++++++++++++++++++++++++++++++++++++++++++++
    logging.info(f"BUCKET: {bucket_name}")
    logging.info(f"S3_FOLDER: {s3_folder}")
    s3_resource = boto3.resource('s3', aws_access_key_id='', 
                                     aws_secret_access_key='')
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        if not os.path.exists(f'model/{os.path.dirname(obj.key)}'):
            os.makedirs(f'model/{os.path.dirname(obj.key)}')
        bucket.download_file(obj.key, f'model/{obj.key}')

    

def convert_doccano_fomart_to_spacy(argv=None):
   # create temp folder data
    import tempfile
    with tempfile.TemporaryDirectory() as data:
        print('created temporary directory', data)

    data = download_s3_file('spacy-ner-test-bucket',known_args.input_dir)
    
    with open('/data/ner_data/test_data.json1', 'rb') as f:
        data = f.readlines()
        print(data)

    testing_data = []
    for record in data:
        entities = []
        read_record = json.loads(record)
        text = read_record['text']
        entities_record = read_record['labels']

        for start, end, label in entities_record:
            entities.append((start, end, label))

        testing_data.append((text, {"entities": entities}))

    return testing_data

# create temp dir model to store model
with tempfile.TemporaryDirectory() as model:
        print('created temporary directory', model)

# download model
download_s3_folder('spacy-ner-test-bucket', 'ner_model')

# load model from disk or GCS. We can't load files from GCS 
nlp = spacy.load(known_args.model_dir)

#test = known_args.input_dir#'test_data.json1'
test = convert_doccano_fomart_to_spacy()


def test_spacy(argv=None):
    #test the model and evaluate it
    examples = test
    tp=0
    tr=0
    tf=0

    ta=0
    c=0        

    #create temp dir result to store result
    with tempfile.TemporaryDirectory() as result:
        print('created temporary directory', result)

    for text,annot in examples:
      f=open(os.path.join(known_args.output_dir+"result"+str(c)+".txt"), 'w')
      
      doc_to_test=nlp(text)
      d={}
      for ent in doc_to_test.ents:
          d[ent.label_]=[]
      for ent in doc_to_test.ents:
          d[ent.label_].append(ent.text)

      for i in set(d.keys()):

          f.write("\n\n")
          f.write(i +":"+"\n")
          for j in set(d[i]):
              f.write(j.replace('\n','')+"\n")
      d={}
      for ent in doc_to_test.ents:
          d[ent.label_]=[0,0,0,0,0,0]
      for ent in doc_to_test.ents:
          doc_gold_text= nlp.make_doc(text)
          gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
          y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
          y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
          if(d[ent.label_][0]==0):
              (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
              a=accuracy_score(y_true,y_pred)
              d[ent.label_][0]=1
              d[ent.label_][1]+=p
              d[ent.label_][2]+=r
              d[ent.label_][3]+=f
              d[ent.label_][4]+=a
              d[ent.label_][5]+=1
      c+=1
    for i in d:
      print("\n For Entity "+i+"\n")
      print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
      print("Precision : "+str(d[i][1]/d[i][5]))
      print("Recall : "+str(d[i][2]/d[i][5]))
      print("F-score : "+str(d[i][3]/d[i][5]))
      
    upload_folder_to_s3(s3bucket, "/result", "result")

Path(known_args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(known_args.output_model_path_file).write_text(known_args.output_dir)

if __name__ == '__main__':
   test_spacy()
