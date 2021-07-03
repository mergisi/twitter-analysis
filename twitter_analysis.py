import azure.functions as func

import pandas as pd
from azure.storage import CloudStorageAccount
from azure.storage.blob import PublicAccess
import logging
import time
import io
import re

start = time.time()

from datetime import datetime
from datetime import timedelta

import json
from pandas.io.json import json_normalize

import spacy

# set environment variables
AZURE_STORAGE_ACCOUNT = AZURE_STORAGE_ACCOUNT
AZURE_STORAGE_KEY = AZURE_STORAGE_KEY
AZURE_SENTIMENT_LIMIT = 500
TENANT = TENANT # target folder and file name
TWEETS_CONTAINER = TWEETS_CONTAINER  # tweets download folder
ANALYSIS_CONTAINER = ANALYSIS_CONTAINER  # PowerBI source folder
DOWNLOAD_BLOB_LIMIT = 5000  # Should be max 10000
TENANT_LEADS_FOLDER = TENANT_LEADS_FOLDER  # #save leads for dublicate check
TENANT_PARAMETER_CONTAINER = TENANT_PARAMETER_CONTAINER
TENANT_PARAMETER_FILE = TENANT_PARAMETER_FILE
MODEL_REGISTER_FILE = MODEL_REGISTER_FILE

norm = lambda x: str(x) if x >= 10 else '0' + str(x)

def main(mytimer: func.TimerRequest) -> None:

    try:
        storage_client = CloudStorageAccount(AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY)
        blob_service = storage_client.create_block_blob_service()
    except Exception as e:
        logging.error("Error while creating Azure blob service : {}".format(e))


    def create_storage_container(storage_container):
        """
        creating storage container
        :param storage_container: name of the storage container
        :return:
        """
        try:
            blob_service.create_container(
                storage_container, public_access=PublicAccess.Blob
            )
            logging.info("Container -{}- is created".format(storage_container))
        except Exception as e:
            logging.error("Error while creating storage container : {}".format(e))


    def check_storage_container_exist(storage_container):
        """Checking an Azure storage container is exist in the storage account.
        :param storage_container:
        :return:
        """
        try:
            status = blob_service.exists(container_name=storage_container)

        except Exception as e:
            logging.error("Error while checking storage container {} : {}".format(storage_container, e))
            raise

        return status


    def check_storage_file_exist_old(storage_container, blob_name):
        """checking an Azure storage container is exist in the storage account.
        :param storage_container:
        :return:
        """
        try:
            generator = blob_service.list_blobs(storage_container)
            storage_blob_list = [blob.name for blob in generator]
        except Exception as e:
            logging.error("Error while getting blob list from Azure storage container -{}-".format(storage_container))
            raise

        if blob_name in storage_blob_list:
            logging.info("Storage blob file -{}- exists".format(blob_name))
            status = True
        else:
            logging.info("Storage blob file -{}- does not exist".format(blob_name))
            status = False
        return status


    if check_storage_container_exist(ANALYSIS_CONTAINER) is False:
        create_storage_container(ANALYSIS_CONTAINER)

    now = datetime.now() - timedelta(hours=4)  # default should be set to 4
    LAST_YEAR = str(now.year)
    LAST_MONTH = str(norm(now.month))
    LAST_DAY = str(norm(now.day))
    LAST_HOUR = now.strftime("%H")
    LAST_MIN = str(now.minute)
    generator = blob_service.list_blobs(TENANT, delimiter='/',
                                        prefix="{}/{}/{}/{}/".format(LAST_YEAR, LAST_MONTH, LAST_DAY,
                                                                     LAST_HOUR))

    # for counting total number of files downloaded.
    data_counter = []
    for blob in generator:
        data_counter.append(blob.name)
        print(blob.name)

    print("Download started... \nETL Date: {}{} \nNumber of blob files: {}\n.\n.\n.".format(now.strftime("%b %d %Y %H"),
                                                                                            ":00", str(len(data_counter))))

    if len(data_counter) <= DOWNLOAD_BLOB_LIMIT:
        randomlist = data_counter
    else:
        randomlist = data_counter[:DOWNLOAD_BLOB_LIMIT]

    if len(randomlist) > 0:
        print("Tweet analysis is started")
        i = 0
        dataloaded = []
        for blob in randomlist:
            i += 1
            print('file name: {} - number: {}'.format(blob, i))
            loader = blob_service.get_blob_to_text(container_name=TWEETS_CONTAINER, blob_name=blob)
            trackerstatusobjects = loader.content.split('\n')
            for trackerstatusobject in trackerstatusobjects:
                dataloaded.append(json.loads(trackerstatusobject))

        df = pd.io.json.json_normalize(dataloaded)

        df['created_at'] = pd.to_datetime(df['created_at'])

        df['text'] = df['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0]).str.replace('[^\w\s]', ' ')

        df['user.descclean'] = df['user.description'].apply(lambda x: re.split('https:\/\/.*', str(x))[0]).str.replace(
            '[^\w\s]', ' ')


        # -------------------------find exist leads---------------------

        loadleads = blob_service.list_blobs(ANALYSIS_CONTAINER, delimiter='/', prefix=TENANT_LEADS_FOLDER + "/")

        dataleads = []
        a = 0
        for blob in loadleads:
            a += 1
            print('file name: {} - number: {}'.format(blob.name, a))
            loader = blob_service.get_blob_to_text(container_name=ANALYSIS_CONTAINER, blob_name=blob.name)
            # split the string str1 with newline.
            datal = loader.content.splitlines()
            dataleads.extend(datal)


        def check_dublicate(user_id):
            if user_id is not None:
                result = [x for x in dataleads if str(x) == str(user_id)]

            if len(result) > 0:
                dublicate = 1
            else:
                dublicate = 0

            return dublicate


        df['exist'] = df['user.id_str'].apply(lambda x: 'True' if check_dublicate(x) == 1 else 'False')
        # print(list(df.columns.values))

        df = df[df['exist'] == 'False']

        # -------------------------find relevant--------------------

        loadparameter = blob_service.get_blob_to_text(container_name=TENANT_PARAMETER_CONTAINER,
                                                      blob_name=TENANT_PARAMETER_FILE, encoding='utf-8')

        # split the string str1 with newline.
        arr1 = loadparameter.content.splitlines()

        # -------------------------find AI model--------------------


        """
        # MLFLOW_MODEL_REGISTER_ID = 'f15f1a65ef624576a7b8418572afa8c4'
        
        loadmodelparameter = blob_service.get_blob_to_text(container_name=TENANT_PARAMETER_CONTAINER,
                                                           blob_name=MODEL_REGISTER_FILE, encoding='utf-8')

        # split the string str1 with newline.
        arr2 = loadmodelparameter.content.splitlines()

        if len(arr2) > 0:
            MLFLOW_MODEL_REGISTER_ID = arr2[0]

        model_uri = f'runs:/{MLFLOW_MODEL_REGISTER_ID}/model'
        """
        
        mymodelpath ="/home/site/wwwroot/TimerTrigger/modelx"
        nlp2 = spacy.load(mymodelpath)

        def check_relevant_ai(profile):
            airesult = 0.0
            if profile is not None:
                doc2 = nlp2(profile)
                a = doc2.cats
                if a['RELEVANT'] > 0.80 and a['NOT RELEVANT'] < 0.30:
                    #print("RELEVANT:", "{:.2f}".format(a['RELEVANT'] * 100))
                    airesult = "{:.2f}".format(a['RELEVANT'] * 100)
                else:
                    #print("NOT RELEVANT", "{:.2f}".format(100 - (a['NOT RELEVANT'] * 100)))
                    airesult = "{:.2f}".format(100 - (a['NOT RELEVANT'] * 100))

            return airesult


        def check_relevant(profile):
            if profile is not None:
                result = [x for x in arr1 if x.lower() in profile.lower()]

            if len(result) > 0:
                relevant = 1
            else:
                relevant = 0
            return relevant


        df['relevant'] = df['user.descclean'].apply(lambda x: 'True' if check_relevant(x) == 1 else 'False')
        df = df[df['relevant'] == 'True']

        # ------------------------AI--------------------------------------------------------

        df['ai_lead_score'] = df['user.description'].apply(lambda x: check_relevant_ai(x))

        # ---------------------------------WRITE TO CSV------------------------------------------#

        df = df[['created_at', 'user.id_str', 'id_str', 'text', 'user.screen_name', \
                 'user.description', 'user.location', 'user.followers_count', 'user.friends_count', 'relevant', 'exist',
                 'ai_lead_score']]


        output = io.StringIO()
        output = df.to_json(orient="records")

        df2 = df['user.id_str']
        output2 = io.StringIO()
        output2 = df2.to_csv(index=False, header=False)

        blob_service.create_blob_from_text(container_name=ANALYSIS_CONTAINER,
                                           blob_name='{}\str-{}-key-0001-{}-{}-{}-{}.json'.format(TENANT, TENANT,
                                                                                                  LAST_YEAR, LAST_MONTH,
                                                                                                  LAST_DAY, LAST_HOUR),
                                           text=output)

        # save leads for further double check control
        blob_service.create_blob_from_text(container_name=ANALYSIS_CONTAINER,
                                           blob_name='{}\str-{}-key-0001-{}-{}-{}-{}.txt'.format(TENANT_LEADS_FOLDER,
                                                                                                 TENANT,
                                                                                                 LAST_YEAR, LAST_MONTH,
                                                                                                 LAST_DAY, LAST_HOUR),
                                           text=output2)

elapsed_time = (time.time() - start)
print("Execution Time {} second".format(elapsed_time))
