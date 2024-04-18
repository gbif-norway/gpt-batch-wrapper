#%%
import os
import time
import logging
import pandas as pd
import requests
import json_repair
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_delay, retry_if_exception_type


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BatchNotReadyException(Exception):
    pass

class OCRDataProcessor:
    def __init__(self, ocr_data, system_prompt, model="gpt-3.5-turbo"):
        self.ocr_data = ocr_data
        self.system_prompt = system_prompt
        self.model = model
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()
        logging.info("OCRDataProcessor instance created with provided OCR data and system prompt.")

    def prepare_data(self):
        df = pd.DataFrame.from_dict(self.ocr_data, orient='index', columns=['ocr']).reset_index()
        df.columns = ['custom_id', 'ocr']
        df['method'] = 'POST'
        df['url'] = '/v1/chat/completions'
        df['body'] = df['ocr'].apply(lambda x: {"model": self.model, "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": x}]})
        logging.info("Dataframe prepared for processing.")
        return df.drop('ocr', axis=1)

    def create_batch_file(self, df):
        batch_file = '/app/batch.jsonl'
        df.to_json(batch_file, orient='records', lines=True)
        logging.info(f"Batch file created at {batch_file}.")
        return batch_file

    def create_batch(self, batch_file):
        oai_file = self.client.files.create(file=open(batch_file, "rb"), purpose="batch")
        logging.info(f"Batch file uploaded with ID: {oai_file.id}")
        return self._create_batch(oai_file.id)

    def _create_batch(self, input_file_id):
        url = "https://api.openai.com/v1/batches"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
        response = requests.post(url, headers=headers, json=data)
        logging.info(f"Batch creation requested for file ID: {input_file_id}. Status: {response.status_code}")
        return response.json()

    def get_batch_details(self, batch_id):
        url = f"https://api.openai.com/v1/batches/{batch_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers)
        logging.info(f"Batch details retrieved for batch ID: {batch_id}. Status: {response.status_code}")
        return response.json()

    @retry(wait=wait_exponential(multiplier=1, max=1800),  # max wait 1800 seconds (30 mins)
        stop=stop_after_delay(86400),  # stop after 24 hours
        retry=retry_if_exception_type(BatchNotReadyException))
    def wait_for_batch_completion(self, batch_id):
        logging.info(f"Checking batch status. Batch ID: {batch_id}")
        batch_details = self.get_batch_details(batch_id)
        if batch_details['status'] in ['completed', 'failed']:
            logging.info(f"Batch processing completed with status: {batch_details['status']}. Batch ID: {batch_id}")
            return batch_details
        else:
            raise BatchNotReadyException("Batch not ready yet, rechecking...")

    def process_responses(self, output_file_id):
        content = self.client.files.content(output_file_id)
        output_file = '/app/output.jsonl'
        content.write_to_file(output_file)
        df = pd.read_json(output_file, lines=True)
        df['json_text'] = df['response'].apply(lambda x: x['body']['choices'][0]['message']['content'])
        df['json'] = df['json_text'].apply(lambda x: json_repair.loads(x))
        logging.info(f"Responses processed and JSON data extracted. Output file: {output_file}")
        return df[['custom_id', 'json']].set_index('custom_id')['json'].to_dict()

#%%
# Usage example
if __name__ == '__main__':
    system_prompt = '''I am an adept herbarium digitization system, working on OCR text extracted from the images of scanned herbarium specimens. First I correct any obvious OCR errors, and then I extract ONLY the following Darwin Core terms:

    - scientificName: Full scientific name, not containing identification qualifications.
    - catalogNumber: Unique identifier for the record in the dataset or collection.
    - recordNumber: Identifier given during recording, often linking field notes and Occurrence record.
    - recordedBy: List of people, groups, or organizations responsible for recording the original Occurrence.
    - year: Four-digit year of the Event.
    - month: Integer for the month of the Event.
    - day: Integer for the day of the Event, not populated unless month and year are filled in.
    - dateIdentified: Date when the subject was determined to represent the Taxon.
    - identifiedBy: Person, group, or organization assigning the Taxon to the subject.
    - verbatimIdentification: Taxonomic identification as it appeared in the original record.
    - country: Name of the country or major administrative unit for the Location.
    - countryCode: Standard code for the country of the Location.
    - decimalLatitude: Geographic latitude in decimal degrees of the Location's center.
    - decimalLongitude: Geographic longitude in decimal degrees of the Location's center.
    - location: A spatial region or named place.
    - minimumElevationInMeters: The lower limit of the range of elevation in meters.
    - maximumElevationInMeters: The upper limit of the range of elevation in meters.
    - verbatimElevation: The original description of the elevation.
    - verbatimCoordinates: The original coordinates. can be MGRS coordinates (f.eks.: UH 68,14).

    If there are multiple valid values for a term, I separate them with "|". If I can't identify information for a specific term, and/or the term is blank, I skip the term in my response. I respond in minified JSON.'''


    ocr_data =    {'d537a581-9d79-431a-99ba-95d0a9a3cc7a':'''Herb . Univers . Osloënsis Imaged 2015
    9102 06 беу CON a6aer'y ( 2 ! W ) թլ ! այ sse ա n ! բյոզլ O L2999-8
    Herb . O. Osloensis Thuidium assimile ( Mitt . ) A.Jaeger NORWAY : ØSTFOLD : FREDRIKSTAD Fredrikstad : Lyngholmen i Onsøy i eng mellom busker MGRS WGS84 PL 031,604 Alt.:5 m LatLong WGS84 59,1707 ° N 10,8045 ° E 23. APR 2015 Leg . & det .: Kåre Arnstein Lye 41442 B - 66577 Reg . 29.04 . 2016
    41442''' ,
        '46448ab7-4d2c-48ab-bdab-22b0af907551':'''Herb . Univers . Osloënsis Imaged 2015
    9102 +062 Bay бемчос xe ' qole | qo | ansadni աոզ ! 100 CONVEN O 92999-8
    MGRS PL 029,604 Alt.:5 m WGS84 LatLong WGS84 59,1707 ° N 10,801 ° E Herb . O. Osloensis Orthotrichum rupestre Schleich . ex Schwägr . NORWAY : ØSTFOLD : FREDRIKSTAD Fredrikstad : Lyngholmen i Onsøy på berg 23. APR 2015 Leg . & det .: Kåre Arnstein Lye 41441 B - 66576 Reg . 29.04 . 2016
    41441
    2.42'''}

    processor = OCRDataProcessor(ocr_data, system_prompt)
    df = processor.prepare_data()
    batch_file = processor.create_batch_file(df)
    batch = processor.create_batch(batch_file)

    try:
        batch_details = processor.wait_for_batch_completion(batch['id'])
    except Exception as e:
        logging.error(f"Failed to complete batch: {str(e)}")

    if batch_details['status'] == 'completed':
        results = processor.process_responses(batch_details['output_file_id'])
        print(results)

    # %%
