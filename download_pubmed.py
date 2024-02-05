import requests
import gzip
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring
import json

def downloadpubmed():
    url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n0001.xml.gz"
    filename = url.split("/")[-1]  # Extract the last part of the URL to get the filename
    unzipped_filename = filename.replace(".gz", "")  # The expected unzipped filename

    # Download the gzipped file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error if the download fails

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Unzip the downloaded file
    with gzip.open(filename, 'rb') as f_in:
        with open(unzipped_filename, 'wb') as f_out:
            f_out.write(f_in.read())

    # Read and print the first line of the XML file
    with open(unzipped_filename, 'r', encoding="utf-8") as f:
        first_line = f.readline()
        print(first_line)


    # Optionally, if you want to delete the original gzipped file after unzipping
    os.remove(filename)

    print(f"Downloaded and unzipped to {unzipped_filename}")


def read_pubmed():
    # Load and parse the XML
    tree = ET.parse('pubmed23n0001.xml')
    root = tree.getroot()

    articles_data = []

    # Iterate through articles
    for article in root.findall('.//PubmedArticle'):
        # Check if the article has an Abstract and PMID
        if article.find('.//Abstract') is not None and article.find(".//PMID") is not None:
            title_element = article.find(".//ArticleTitle")
            abstract_element = article.find(".//Abstract/AbstractText")
            pmid_element = article.find(".//PMID")
            mesh_headings = article.findall(".//MeshHeading/DescriptorName")
            keywords = article.findall(".//KeywordList/Keyword")
            pub_date = article.find(".//PubDate/Year")
            iso_abbreviation = article.find(".//Journal/ISOAbbreviation")
            substances = [substance.text for substance in article.findall(".//ChemicalList/Chemical/NameOfSubstance")]

            # Extract the text from each element
            title_text = title_element.text if title_element is not None else ""
            abstract_text = abstract_element.text if abstract_element is not None else ""
            pmid_text = pmid_element.text if pmid_element is not None else ""
            mesh_headings_text = ", ".join([mh.text for mh in mesh_headings])
            keywords_text = ", ".join([kw.text for kw in keywords])
            pub_date_text = pub_date.text if pub_date is not None else ""
            iso_abbreviation_text = iso_abbreviation.text if iso_abbreviation is not None else ""

            # Concatenate title and abstract and include PMID
            combined_text = title_text + " " + abstract_text
            articles_data.append({'text': combined_text,
                                  'PMID': pmid_text,
                                  'mesh_headings': mesh_headings_text,
                                  'keywords': keywords_text,
                                  'pub_date': pub_date_text,
                                  'iso_abbreviation_text': iso_abbreviation_text,
                                  'substances': substances})
            break

    # Store the concatenated information in a JSON file
    with open("articles_data.json", "w") as f:
        json.dump(articles_data, f, indent=4)

def read_pubmed_json():
    with open("articles_data.json", "r") as f:
        data = json.load(f)
        print(len(data))
        # Assuming you want to print the first 3 articles as separate lines
        for article in data[:3]:
            print(json.dumps(article, indent=4))


if __name__ == "__main__":
    # downloadpubmed()
    read_pubmed()
    # read_pubmed_json()