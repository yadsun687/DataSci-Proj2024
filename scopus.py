import requests
import pandas as pd

# Replace with your Scopus API key
API_KEY = "f5624674690906ad58405769d1c9b433"
BASE_URL = "https://api.elsevier.com/content/search/scopus"
headers = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json"
}

# Function to get journal metadata
def get_journal_metadata(issn, api_key):
    url = f"https://api.elsevier.com/content/serial/title/issn/{issn}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "serial-metadata-response" in data:
            entries = data["serial-metadata-response"].get("entry", [])
            if entries:
                return data
            else:
                print(f"No journal metadata found for ISSN: {issn}")
        else:
            print("Invalid response format.")
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Set up parameters for the request
params = {
    "query": "all(gene)",  # Query
    "count": 25,            # Number of results to retrieve per request
    "start": 0,             # Start index
    "view": "STANDARD"      # View type
}

all_data = []

# Loop to extract 1000 records (40 requests of 25 results each)
for start_index in range(0, 1000, 25):
    params["start"] = start_index
    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()

        # Extract entries from the response
        entries = data.get("search-results", {}).get("entry", [])

        for entry in entries:
            issn = entry.get("prism:issn", "N/A")
            journal_data = get_journal_metadata(issn, API_KEY)

            # Extract subject areas if available in the response from get_journal_metadata
            subject_area_list = []
            if journal_data and isinstance(journal_data, dict):
                metadata_response = journal_data.get("serial-metadata-response", {})
                metadata_entries = metadata_response.get("entry", [])
                
                if metadata_entries:
                    subject_areas = metadata_entries[0].get("subject-area", [])
                    subject_area_list = [
                        area.get("$", "Unknown") for area in subject_areas
                    ]

            # Collect data from the entry and append to the list
            all_data.append({
                "date_delivered_year": entry.get("prism:coverDate", "Unknown")[:4],
                "date_sort_year": entry.get("prism:coverDate", "Unknown")[:4],
                "author_group": entry.get("author_group", "Unknown"),
                "citation_title": entry.get("dc:title", "Unknown"),
                "affiliation_country": entry.get("affiliation", [{}])[0].get("affiliation-country", "Unknown"),
                "affiliation_city": entry.get("affiliation", [{}])[0].get("affiliation-city", "Unknown"),
                "affiliation_organization": entry.get("affiliation", [{}])[0].get("affilname", "Unknown"),
                "corresponding_author_given_name": entry.get("corresponding_author_given_name", "Unknown"),
                "corresponding_author_surname": entry.get("corresponding_author_surname", "Unknown"),
                "corresponding_author_indexed_name": entry.get("dc:creator", "Unknown"),
                "citation_language": entry.get("citation_language", "Unknown"),
                "source_country": entry.get("affiliation", [{}])[0].get("affiliation-country", "Unknown"),
                "source_publication_year": entry.get("prism:coverDate", "Unknown")[:4],
                "source_publisher_name": entry.get("source_publisher_name", "Unknown"),
                "classificationgroup": entry.get("classificationgroup", "Unknown"),
                "dbcollection": entry.get("dbcollection", "Unknown"),
                "ref_count": entry.get("ref_count", "Unknown"),
                "reference": entry.get("reference", "Unknown"),
                "affiliation": entry.get("affiliation", "Unknown"),
                "coverDate": entry.get("prism:coverDate", "Unknown"),
                "aggregationType": entry.get("prism:aggregationType", "Unknown"),
                "author": entry.get("dc:creator", "Unknown"),
                "publicationName": entry.get("prism:publicationName", "Unknown"),
                "citedby_count": entry.get("citedby-count", "Unknown"),
                "title": entry.get("dc:title", "Unknown"),
                "publisher": entry.get("prism:publicationName", "Unknown"),
                "subject_area": ", ".join(subject_area_list) if subject_area_list else "Unknown",
                "abstract_language": entry.get("abstract_language", "Unknown"),
            })

    else:
        print(f"Error: {response.status_code} - {response.text}")

# Create DataFrame and save as CSV
df = pd.DataFrame(all_data)
df.to_csv("scopus_data_1000.csv", index=False)
