def train(argv=None):
    import json
    import plac
    import random
    import warnings
    import argparse
    from pathlib import Path
    import spacy
    import logging
    import boto3
    import botocore
    import os
    from spacy.util import minibatch, compounding
    from spacy.gold import GoldParse
    from spacy.scorer import Scorer
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--output-model-path-file', help='')
    known_args, _ = parser.parse_known_args(argv)
    
    # create temp directory data
    with tempfile.TemporaryDirectory() as data:  
        print('created temporary directory', data)


    def train_spacy(argv=None):
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
        download_s3_file('spacy-ner-test-bucket', known_args.input_dir)
    
        with open(f'/data/data/training_data.txt', 'r') as f:
            TRAIN_DATA = json.load(f)
            print(TRAIN_DATA)

        nlp = spacy.blank('en') 
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)
        

        for _,annotations in TRAIN_DATA:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes): 
            optimizer = nlp.begin_training()
            for itn in range(known_args.iteration):
                print("Starting iteration " + str(itn))
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in TRAIN_DATA:
                    try:
                        nlp.update(
                            [text],  
                            [annotations],  
                            drop=0.2,  
                            sgd=optimizer,  
                            losses=losses)
                    except Exception as error:
                        print(error)
                        continue
                print(losses)
        
        return nlp
    trainer = train_spacy()

    # create temp directory model
    with tempfile.TemporaryDirectory() as model:  
        print('created temporary directory', model)

    trainer.to_disk(known_args.output_dir)

    # upload saved model folder to s3
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
    upload_folder_to_s3(s3bucket, "/model", "ner_model")
  

    Path(known_args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(known_args.output_model_path_file).write_text(known_args.output_dir)
    

if __name__ == '__main__':
    train()