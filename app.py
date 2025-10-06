from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import pickle
import io
import os
import subprocess
import tempfile
import time

app = Flask(__name__)

# ----------------------------
# Load Saved Models
# ----------------------------
with open('Regressor_model.kpl', 'rb') as f:
    regression_model = pickle.load(f)

with open('classification_model.kpl', 'rb') as f:
    classification_model = pickle.load(f)

with open('scalar.kpl', 'rb') as f:
    scaler_model = pickle.load(f)


# ----------------------------
# Utility Functions
# ----------------------------
def run_padel_descriptor(input_smi, output_csv, padel_jar='PaDEL-Descriptor/PaDEL-Descriptor.jar', fingerprints=True):
    """Run PaDEL-Descriptor on input .smi and return DataFrame."""
    if not os.path.exists(input_smi):
        raise FileNotFoundError(f"SMILES file not found: {input_smi}")
    if not os.path.exists(padel_jar):
        raise FileNotFoundError(f"PaDEL jar not found: {padel_jar}")

    cmd = [
        "java", "-Xms2G", "-Xmx4G", "-jar", padel_jar,
        "-2d",
        "-dir", os.path.dirname(input_smi),
        "-file", output_csv
    ]
    if fingerprints:
        cmd.append("-fingerprints")

    print("Running PaDEL-Descriptor... please wait...")
    subprocess.run(cmd, check=True)
    print("PaDEL finished generating descriptors")

    # Load CSV into memory
    desc_df = pd.read_csv(output_csv)
    return desc_df


def calculate_descriptors(df):
    """Generating molecular descriptors from SMILES using temporary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        smi_file = os.path.join(tmpdir, 'input.smi')
        out_file = os.path.join(tmpdir, 'descriptors.csv')

        # Save SMILES to .smi file (PaDEL input)
        df[['SMILES']].to_csv(smi_file, index=False, header=False)

        # Run PaDEL
        desc_df = run_padel_descriptor(input_smi=smi_file, output_csv=out_file)

        # Files auto-deleted when tempdir closes 

    return desc_df


def prepare_features(df):
    """Extract model input columns (should match your trained features)."""
    # TODO: Replace this list with your actual feature columns used in model training
    feature_columns = ['Descriptor_1', 'Descriptor_2', 'Descriptor_3', 'Descriptor_4', 'Descriptor_5']

    # Use intersection to avoid KeyErrors
    available_features = [col for col in feature_columns if col in df.columns]
    return df[available_features]


# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


from flask import jsonify

@app.route('/predict', methods=['POST'])
def predict():
    """Handle CSV upload or manual input, then compute descriptors and predict."""
    global result_file

    smiles_input = request.form.get('smiles_input', '').strip()
    file = request.files.get('file')

    if not file and not smiles_input:
        return jsonify({"status": "error", "message": "Please upload a CSV or enter a SMILES string."})

    # Preparing Inputted Data
    if smiles_input:
        df = pd.DataFrame({'SMILES': [smiles_input]})
    else:
        df = pd.read_csv(file)

    original_data_html = df.head(5).to_html(classes='table table-striped', index=False)

    # Calculating the descriptors
    try:
        descriptor_df = calculate_descriptors(df)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Descriptor generation failed: {e}"})

    descriptor_data_html = descriptor_df.head(5).to_html(classes='table table-bordered', index=False)

    # Loading Feature Columns
    try:
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to load feature columns: {e}"})

    # Extracting model input
    try:
        X = descriptor_df[feature_columns]
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Descriptor calculation missing features: {e}"})

    if X.shape[0] == 0 or X.shape[1] == 0:
        return jsonify({"status": "error", "message": "Descriptor calculation returned no usable features."})

    #  Scaling or standardizing the submitted features
    try:
        X_scaled = scaler_model.transform(X)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Feature scaling failed: {e}"})

    # Predicting the submitted molecule
    try:
        reg_preds = np.round(regression_model.predict(X_scaled), 2)
        class_preds = classification_model.predict(X_scaled)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Model prediction failed: {e}"})

    # Combination of  results
    df['Smiles'] = smiles_input
    df['pIC50'] = reg_preds
    df['Remark'] = class_preds
    

    prediction_data_html = df[['Smiles' ,'pIC50', 'Remark']].head(5).to_html(
        classes='table table-success table-bordered', index=False
    )

    # Preparing result CSV
    output = pd.concat([df, descriptor_df], axis=1)
    csv_buffer = io.StringIO()
    output.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    result_file = io.BytesIO()
    result_file.write(csv_buffer.getvalue().encode())
    result_file.seek(0)

    # Returning clean JSON 
    return jsonify({
        "status": "success",
        "original_data": original_data_html,
        "descriptor_data": descriptor_data_html,
        "prediction_data": prediction_data_html,
        "download_ready": True
    })




@app.route('/download')
def download():
    """Allow user to download prediction results."""
    if 'result_file' not in globals():
        return redirect(url_for('home'))
    return send_file(
        result_file,
        as_attachment=True,
        download_name='prediction_results.csv',
        mimetype='text/csv'
    )


if __name__ == '__main__':
    app.run(debug=True)
