from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
import os
import json

from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import mimetypes

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Gradient credentials from user data
os.environ['GRADIENT_ACCESS_TOKEN'] = 'bHhlnhTYj4sNq27cWAw2xj37nfng8Upy'
os.environ['GRADIENT_WORKSPACE_ID'] = 'f0e968f2-6596-466a-b42b-43655767ff38_workspace'

# Load Cassandra credentials
with open("kaustubhkapare@gmail.com-token (1).json") as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]

# Cassandra connection setup
auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cloud_config = {'secure_connect_bundle': 'secure-connect-temp-db-kaust.zip'}
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

# LLM and Embedding setup
llm = GradientBaseModelLLM(
    base_model_slug="llama2-7b-chat",
    max_tokens=400,
)

embed_model = GradientEmbedding(
    gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    gradient_model_slug="bge-large",
)

# Service Context setup
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=256,
)
set_global_service_context(service_context)

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file: FileStorage = request.files['file']
    user_query = request.form['user_query']

    if file and allowed_file(file.filename) and user_query:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check if the uploaded file is a PDF
        mime_type, encoding = mimetypes.guess_type(file_path)
        if mime_type != 'application/pdf':
            os.remove(file_path)
            return render_template('index.html', error="Please upload a valid PDF file.")

        # Reload documents from the updated directory
        updated_documents = SimpleDirectoryReader(app.config['UPLOAD_FOLDER']).load_data()

        # Initialize the index with the updated documents
        index = VectorStoreIndex.from_documents(updated_documents, service_context=service_context)
        query_engine = index.as_query_engine()

        # Generate response using LLM and user query
        response = query_engine.query(user_query)

        # Render the result template with the response
        return render_template('result.html', result=response)

    return render_template('index.html', error="Please upload a PDF file and enter a query.")

if __name__ == '__main__':
    app.run(debug=True)
