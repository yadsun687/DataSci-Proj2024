from kafka import KafkaConsumer
import json
import csv
from datetime import datetime

# Kafka configuration
consumer = KafkaConsumer(
    'journal_updates',
    bootstrap_servers='localhost:9092', 
    auto_offset_reset='earliest', 
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# File setup
current_date = datetime.now().strftime('%d%m%Y')  # ddmmyyyy format
file_name = f"{current_date}.csv"
article_ids = []

for message in consumer:
    article = message.value
    article_id = article["article_id"]
    print(f'Processing article ID: {article_id}')
    article_ids.append([article_id])

    if len(article_ids) >= 100: 
        break

with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["article_id"])
    writer.writerows(article_ids)

print(f"Article IDs saved to {file_name}")
