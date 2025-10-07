from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify  # type: ignore
import pandas as pd
import numpy as np
import pickle
import joblib
import io
import os
import subprocess
import tempfile
import time
import requests
from tempfile import TemporaryDirectory

app = Flask(__name__)


# Loading Saved Models

if not os.path.exists("Regressor_model.pkl"):
    r = requests.get(os.environ['MODEL_DOWNLOAD_URL'])
    open("Regressor_model.pkl", "wb").write(r.content)
regression_model = joblib.load("Regressor_model.pkl")

if not os.path.exists("classification_model.pkl"):
    r = requests.get(os.environ['MODEL_DOWNLOAD_URL'])
    open("classification_model.pkl", "wb").write(r.content)
classification_model = joblib.load("classification_model.pkl")

with open('scalar.pkl', 'rb') as f:
    scaler_model = pickle.load(f)


# Utility Functions

def split_smi_file(input_smi, chunk_size=500):
    """Split a .smi file into smaller chunks of `chunk_size` molecules."""
    with open(input_smi, 'r') as f:
        lines = [line for line in f if line.strip()]
    for i in range(0, len(lines), chunk_size):
        chunk_path = f"{os.path.splitext(input_smi)[0]}_part{i//chunk_size + 1}.smi"
        with open(chunk_path, 'w') as out:
            out.writelines(lines[i:i + chunk_size])
        yield chunk_path


def run_padel_descriptor(input_smi, output_csv, padel_dir=None, fingerprints=True):
    """Run PaDEL-Descriptor on possibly large .smi file by chunking automatically."""
    if padel_dir is None:
        padel_dir = os.path.join(os.getcwd(), 'PaDEL-Descriptor')

    padel_jar = os.path.join(padel_dir, 'PaDEL-Descriptor.jar')
    lib_dir = os.path.join(padel_dir, 'lib', '*')

    if not os.path.exists(input_smi):
        raise FileNotFoundError(f"SMILES file not found: {input_smi}")
    if not os.path.exists(padel_jar):
        raise FileNotFoundError(f"PaDEL jar not found: {padel_jar}")

    all_results = []
    with TemporaryDirectory() as tempdir:
        for chunk_file in split_smi_file(input_smi, chunk_size=500):
            chunk_output = os.path.join(tempdir, f"{os.path.basename(chunk_file)}.csv")

            cmd = [
                "java", "-Xms128M", "-Xmx256M",
                "-cp", f"{padel_jar}:{lib_dir}",
                "padeldescriptor.PaDELDescriptorApp",
                "-2d",
                "-dir", os.path.dirname(chunk_file),
                "-file", chunk_output
            ]
            if fingerprints:
                cmd.append("-fingerprints")

            print(f"Running PaDEL on {os.path.basename(chunk_file)} ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"PaDEL failed on chunk {chunk_file}")

            df = pd.read_csv(chunk_output)
            all_results.append(df)

            # This will Clean up to save memory space
            os.remove(chunk_file)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print("✅ PaDEL finished generating all descriptors successfully.")
    return final_df


def calculate_descriptors(df):
    """Generating molecular descriptors from SMILES using temporary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        smi_file = os.path.join(tmpdir, 'input.smi')
        out_file = os.path.join(tmpdir, 'descriptors.csv')

        # Saving SMILES to .smi file (PaDEL input)
        df[['SMILES']].to_csv(smi_file, index=False, header=False)

        # Runing PaDEL (memory-optimized version)
        desc_df = run_padel_descriptor(input_smi=smi_file, output_csv=out_file)

    return desc_df


def prepare_features(df):
    """Extract model input columns (should match your trained features)."""
    feature_columns = ['Descriptor_1', 'Descriptor_2', 'Descriptor_3', 'Descriptor_4', 'Descriptor_5']
    available_features = [col for col in feature_columns if col in df.columns]
    return df[available_features]


# Flask Routes

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle CSV upload or manual input, then compute descriptors and predict."""
    global result_file

    smiles_input = request.form.get('smiles_input', '').strip()
    file = request.files.get('file')

    if not file and not smiles_input:
        return jsonify({"status": "error", "message": "Please upload a CSV or enter a SMILES string."})

    if smiles_input:
        df = pd.DataFrame({'SMILES': [smiles_input]})
    else:
        df = pd.read_csv(file)

    original_data_html = df.head(5).to_html(classes='table table-striped', index=False)

    try:
        descriptor_df = calculate_descriptors(df)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Descriptor generation failed: {e}"})

    descriptor_data_html = descriptor_df.head(5).to_html(classes='table table-bordered', index=False)

    try:
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to load feature columns: {e}"})

    try:
        X = descriptor_df[feature_columns]
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Descriptor calculation missing features: {e}"})

    if X.shape[0] == 0 or X.shape[1] == 0:
        return jsonify({"status": "error", "message": "Descriptor calculation returned no usable features."})

    try:
        X_scaled = scaler_model.transform(X)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Feature scaling failed: {e}"})

    try:
        reg_preds = np.round(regression_model.predict(X_scaled), 2)
        class_preds = classification_model.predict(X_scaled)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Model prediction failed: {e}"})

    # Debugging log, though it is optional but it will helps on Render
    print(f"DEBUG: len(df)={len(df)}, len(reg_preds)={len(reg_preds)}, len(class_preds)={len(class_preds)}")

    # This ensure that prediction lengths match dataframe rows
    pred_len = len(reg_preds)
    df_len = len(df)

    if pred_len != df_len:
        print("⚠️ Length mismatch detected between input and predictions. Adjusting...")
        min_len = min(pred_len, df_len)
        reg_preds = reg_preds[:min_len]
        class_preds = class_preds[:min_len]
        df = df.iloc[:min_len]

    #Assigning predictions safely
    df['Smiles'] = smiles_input if len(df) == 1 else df['SMILES']
    df['pIC50'] = reg_preds
    df['Remark'] = class_preds


    prediction_data_html = df[['Smiles', 'pIC50', 'Remark']].head(5).to_html(
        classes='table table-success table-bordered', index=False
    )

    output = pd.concat([df, descriptor_df], axis=1)
    csv_buffer = io.StringIO()
    output.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    result_file = io.BytesIO()
    result_file.write(csv_buffer.getvalue().encode())
    result_file.seek(0)

    return jsonify({
        "status": "success",
        "original_data": original_data_html,
        "descriptor_data": descriptor_data_html,
        "prediction_data": prediction_data_html,
        "download_ready": True
    })

@app.route('/download')
def download():
    if 'result_file' not in globals():
        return redirect(url_for('home'))
    return send_file(
        result_file,
        as_attachment=True,
        download_name='prediction_results.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
