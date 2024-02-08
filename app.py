from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import hashlib

app = Flask(__name__)

df = None  # Global variable to hold the DataFrame

# Laplace Mechanism
def laplace_mechanism(value, epsilon):
    return value + np.random.laplace(0, 1 / epsilon)

# Generalization
def generalize_continuous_variable(data, num_bins):
    min_val = data.min()
    max_val = data.max()
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_ranges = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges) - 1)]
    
    def map_to_bin_range(x):
        for low, high in bin_ranges:
            if low <= x < high:
                return f"{low}-{high}"
        return f"{high}-{high}"
    
    generalized_data = data.map(map_to_bin_range)
    return generalized_data

# PCA de-identification
def pca_deidentify(data):
    exclude_columns = []
    
    # Identify columns containing strings
    string_columns = data.select_dtypes(include='object').columns.tolist()

    # Exclude columns containing strings
    exclude_columns += string_columns

    # If no columns need to be masked, return the original data
    if len(exclude_columns) == len(data.columns):
        return data

    # Mask columns using PCA
    columns_to_mask = data.columns.difference(exclude_columns)
    masked_data = data.copy()
    
    for col in columns_to_mask:
        pca = PCA(n_components=1)  # Only one component because we are doing one column at a time
        transformed_data = pca.fit_transform(masked_data[[col]])  # Double brackets to keep it as DataFrame
        masked_data[col] = transformed_data  # Replacing original data with transformed data
    
    return masked_data

#Hashing
def hash_email_local_part(email):
    local_part = email.split('@')[0]  # Get the local part of the email
    domain_part = email.split('@')[1]  # Get the domain part of the email
    hashed_local = hashlib.sha256(local_part.encode()).hexdigest()  # Hash the local part
    return f"{hashed_local}@{domain_part}"  # Return the hashed email


@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/deidentify', methods=['POST'])
def deidentify():
    global df
    uploaded_file = request.files['file']
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()
    
    # Convert the top 5 rows of the dataframe to HTML
    preview_html = df.head().to_html(classes='data', header="true")
    
    return render_template('deidentify.html', columns=columns, preview_html=preview_html)

@app.route('/results', methods=['POST'])
def results():
    global df
    methods = request.form.to_dict()

    for col in df.columns:
        method = methods.get(col, "none")
        if method == 'laplace':
            epsilon = float(methods.get(f"{col}_epsilon", 1))
            df[col] = df[col].apply(lambda x: laplace_mechanism(x, epsilon))
        elif method == 'generalization':
            bins = int(methods.get(f"{col}_bins", 5))  # Default to 5 bins
            df[col] = generalize_continuous_variable(df[col], bins)
        elif method == 'pca':
            if df.shape[1] > 1:  # Only allow PCA if more than one column selected
                df = pd.DataFrame(pca_deidentify(df))
        elif method == 'hashing':
            if col.lower().endswith('email'):  # Check if the column name suggests it contains emails
                df[col] = df[col].apply(hash_email_local_part)
            else:
                df[col] = df[col].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())

    csv_data = df.to_csv(index=False)
    response = make_response(csv_data)
    response.headers.set("Content-Type", "text/csv")
    response.headers.set("Content-Disposition", "attachment; filename=deidentified.csv")
    return response

if __name__ == '__main__':
    app.run(debug=True)
