from datetime import datetime, timedelta
from kafka import KafkaProducer
import requests
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
topic = 'journal_updates'

# Fetch daily journal updates using PubMed
def fetch_daily_updates():
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    date_range = f"{today.strftime('%Y/%m/%d')}:{tomorrow.strftime('%Y/%m/%d')}[PDAT]" 
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {
        'db': 'pubmed',
        'term': date_range,
        'retmode': 'json',
        'retmax': 100  
    }
    response = requests.get(url, params=params)
    return response.json()

# Produce messages to Kafka
def send_updates_to_kafka():
    articles = fetch_daily_updates()
    for article in articles.get('esearchresult', {}).get('idlist', []):
        producer.send(topic, {'article_id': article})
        print(f"Sent article ID: {article}")

send_updates_to_kafka()
