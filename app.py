from flask import Flask, redirect, render_template, jsonify, request, url_for
from anntraining9 import maintrain
from predict import inputGrid
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'uploadfiles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pkl', 'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


grid = [[0]*5 for _ in range(7)]
asciiNumber = 0
asciiCharacter = None


@app.route('/')
def index():
    file_name = 'currentTraining.xlsx'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    # Check if the file exists
    if os.path.exists(file_path):
        file_exists = True
    else:
        file_exists = False
    return render_template('rami-acr.html', file_exists=file_exists, file_name=file_name)

@app.route('/train-model', methods=['POST'])
def train_model():
    epochinput = request.form.get('epoch_training')
    inputrate = request.form.get('learning_rate')
    errorinput = request.form.get('error_training')
    neuronsinput = request.form.get('neurons')

    maintrain(epochinput,inputrate,errorinput,neuronsinput)

    # Assuming you have some function to handle these parameters
    # train_network(neurons, learning_rate, error_training, epoch_training)

    print("Training Parameters:", neuronsinput, inputrate, errorinput, epochinput)
    
    # Redirect or render a template after training
    return redirect(url_for('index'))


@app.route('/click', methods=['POST'])
def click():
    global asciiNumber
    x = int(request.form['x'])
    y = int(request.form['y'])

    grid[x][y] = 1 - grid[x][y]
    rawAscii = inputGrid(grid)
    asciiNumber = round(rawAscii)
    asciiCharacter = chr(asciiNumber)

    return jsonify({'success': True, 'x': x, 'y': y, 'state': grid[x][y], "asciiNumber": asciiNumber, "asciiCharacter": asciiCharacter})


@app.route('/success', methods=['POST'])
def succesUpload():
    if 'file-input' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file-input']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and file.filename.endswith('.pkl'):
        # Define a new filename
        filename = f"currentBiases.pkl"  # Custom filename with a timestamp

        # Path where the file will be saved
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Check if file exists
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove the existing file
            print("Existing file removed, replacing with the new file.")

        # Save the new file
        file.save(file_path)
        return redirect(url_for('gridInput'))
    elif file and file.filename.endswith('.xlsx'):
        # Define a new filename
        filename = f"currentTraining.xlsx"  # Custom filename with a timestamp

        # Path where the file will be saved
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Check if file exists
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove the existing file
            print("Existing file removed, replacing with the new file.")

        # Save the new file
        file.save(file_path)
        return redirect(url_for('index'))
    else:
        return jsonify({'success': False, 'error': 'Invalid file type'})
    

@app.route('/rami')
def ramiACR():
    return render_template('rami-acr.html')


@app.route('/gridInput')
def gridInput():
    global grid, asciiNumber
    asciiNumber = 0
    # Reset the grid every time the page is loaded
    grid = [[0]*5 for _ in range(7)]
    asciiCharacter = None
    file_name = 'currentBiases.pkl'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    # Check if the file exists
    if os.path.exists(file_path):
        file_exists = True
    else:
        file_exists = False
    return render_template('gridInput.html', grid=grid, asciiNumber=asciiNumber, asciiCharacter=asciiCharacter, file_exists=file_exists, file_name=file_name)


@app.route('/get-grid', methods=['GET'])
def getGrid():
    return jsonify({})


if __name__ == '__main__':
    app.run(debug=True)
