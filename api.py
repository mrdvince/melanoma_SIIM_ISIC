import os
from flask import Flask, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'static'


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_file.save(os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            ))
            return render_template('index.html', prediction=1)

    return render_template('index.html', prediction=0)


if __name__ == '__main__':
    app.run(debug=True)
