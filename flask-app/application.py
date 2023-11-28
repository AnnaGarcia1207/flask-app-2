from flask import Flask, render_template, request
from elasticsearch import Elasticsearch, exceptions
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
from PIL import Image
import os

application = Flask(__name__)

# Using config file to stre creds
# config_file_path = "config.json"

# with open(config_file_path, 'r') as file:
#     config = json.load(file)

# CLOUD_ID = config['elasticsearch']['cloud_id']
# ELASTIC_USERNAME = config['elasticsearch']['username']
# ELASTIC_PASSWORD = config['elasticsearch']['password']
# CERT_FINGERPRINT = config['elasticsearch']['cert_fingerprint']

# elastic_search = Elasticsearch(
#         cloud_id=CLOUD_ID,  
#         basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
#         ssl_assert_fingerprint=CERT_FINGERPRINT
#         )

# Another way: Using .env to store creds -------------------------------
# project_folder = os.path.expanduser('/') 
# load_dotenv(os.path.join(project_folder, '.env'))





# Your Elasticsearch operations here
# elastic_search = Elasticsearch(
#     cloud_id= os.getenv('ES_CLOUD_ID'),
#     api_key= os.getenv('ES_API_KEY'),
#     ssl_assert_fingerprint=os.getenv('ES_SSL')
# )

elastic_search = Elasticsearch(
    cloud_id= "dataViz:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbzo0NDMkZmUyZDZlZTk3OTIxNGMyOGEzYWEwYjMyYjU1ZDVjNDYkMjYwYzE4NTUxZmU0NGRhYzhlYmRjNzU0Mjc3Y2QwYmE=", 
    api_key="SnVmR0Q0d0J6R3piQWtsb25MaVo6RTBhdXZwNTlTZWV2eEJUNE0xSjFpdw==", 
    ssl_assert_fingerprint="89:0C:01:EC:62:D3:AB:51:5A:E0:9D:11:3B:2E:5D:74:5A:79:8A:0D:70:2E:7D:13:E6:2F:1D:53:FF:3A:CF:1E" 
    )

# except exceptions as e:
#     # Handling the exception
#     print("An error occurred:", e)
#     try:
#         # Using info() method to get information about the cluster
#         info = elastic_search.info()
#         print("Cluster info:", info)
#     except exceptions as info_error:
#         print("Error fetching cluster info:", info_error)

print(elastic_search.info())

# if not elastic_search.ping():
#     print("EXCEPTION: Check if Elasticsearch is up and running")
#     sys.exit(1) 

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID).to(device)
processor = CLIPProcessor.from_pretrained(model_ID)
tokenizer = CLIPTokenizer.from_pretrained(model_ID)

index_name = "scifig-pilot"

def create_results_map(search_results):
    results = {}
    for hit in search_results:
        
        hit_id = hit['_id']
        score = hit['_score'] *100
        label = hit['_source']['label']
        location = hit['_source']['location']
        name = location.split('/')[-1]

        # hit_id = hit['_id']
        # score = hit['_score'] *100
        # label = hit['_source']['label']
        # location = hit['_source']['image_path']
        # name = location.split('/')[-1]
        # confidence_dict = ast.literal_eval(hit['_source']['confidence'])
        # one = confidence_dict[1]['label']
        # one_score = confidence_dict[1]['score']

        # two = confidence_dict[2]['label']
        # two_score = confidence_dict[2]['score']

        # three = confidence_dict[3]['label']
        # three_score = confidence_dict[3]['score']

        results[hit_id] = {
            "score" : f'{score:.2f}%',
            "name" : name,
            "label" : label,
            "location" : location,
        }
    
    return results

def search_embeddings(embedding_vector, embedding_type):
    # source_fields = ['Id', 'label', 'image_path', 'confidence']
    source_fields = ['id', 'label', 'location']
    k = 99
    num_candidates = 150
    query = {
        "field": embedding_type,
        "query_vector": embedding_vector,
        "k": k,
        "num_candidates": num_candidates
    }
    try:
        response = elastic_search.knn_search(index=index_name, knn=query, source=source_fields)
        # print(response)
        return response['hits']['hits']

    except exceptions.RequestError as e:
        print(f"Error: {e.info['error']['root_cause'][0]['reason']}")
    

def process_text_query(input_query):
   inputs = tokenizer(input_query, return_tensors="pt").to(device)
   text_embeddings = model.get_text_features(**inputs)
   embeddings_as_numpy = text_embeddings.cpu().detach().numpy().reshape(-1)

   # Dictionary containing : ['id', 'label', 'location']
   search_results = search_embeddings(embeddings_as_numpy, "text_embeddings")

   results_map = create_results_map(search_results)

   return results_map

def process_image_query(input):
    input_image = Image.open(input).convert("RGB")
    image = processor(
    text = None,
    images = input_image,
    return_tensors="pt"
    )["pixel_values"].to(device)

    embeddings = model.get_image_features(image)

    numpy_embeddings = embeddings.cpu().detach().numpy().reshape(-1)

    search_results = search_embeddings(numpy_embeddings, "img_embeddings")
    results_dict = create_results_map(search_results)

    return results_dict

   


@application.route('/', methods=['GET','POST'])
def index():
    # grab the text query
    results_dict = None
    if request.method =='POST':
        text_query = request.form.get('searchQuery')

        # passed into the process_text_query and k
        results_dict = process_text_query(text_query)
    
    return render_template("index.html", results=results_dict)


@application.route('/upload', methods=['POST'])
def upload():
    results_dict = None
    if 'imageUpload' in request.files:
        image_query = request.files['imageUpload']

        results_dict = process_image_query(image_query)
    
    return render_template("index.html", results=results_dict)

if __name__=="__main__":
    application.run(debug=True)