def convert_doccano_fomart_to_spacy(argv=None):
    import os
    import json
    import tempfile
    from pathlib import Path
    import argparse
    import logging
    import boto3
    import botocore

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_dir')

    parser.add_argument('--output-model-path-file', help='')
    known_args, _ = parser.parse_known_args(argv)
    #read from gcs bucket

    #create temp directory
    with tempfile.TemporaryDirectory() as data:
        print('created temporary directory', data)

    # download training data from s3 bucket
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
    data = download_s3_file('spacy-ner-test-bucket',known_args.input_path)
    
    with open('/data/ner_data/training_data.json1', 'rb') as f:
        data = f.readlines()
        print(data)
        
        
    training_data = []
    for record in data:
        entities = []
        read_record = json.loads(record)
        text = read_record['text']
        entities_record = read_record['labels']

        for start, end, label in entities_record:
            entities.append((start, end, label))

        training_data.append((text, {"entities": entities}))
    
    with open(known_args.output_dir, 'w') as f:
        json.dump(training_data, f)

    #upload processed file to s3 data directory    
    def upload_file_to_s3_bucket(bucket_name, local_dir, category_folder, file_key):
        s3_client = boto3.client('s3', aws_access_key_id='', 
                                 aws_secret_access_key='')
        file_path = f'{local_dir}/{file_key}'
        print("uploading files")
        s3_client.upload_file(file_path, bucket_name, os.path.join(category_folder, file_key))

    upload_file_to_s3_bucket('spacy-ner-test-bucket', '/data', 'data', 'training_data.txt')

    Path(known_args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(known_args.output_model_path_file).write_text(known_args.output_dir)

if __name__ == '__main__':
    convert_doccano_fomart_to_spacy()
