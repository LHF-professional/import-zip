import os

from flask import Flask, render_template, jsonify, request
import time
from random import randrange

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        a = request.form.get('firstInput')
        b = request.form.get('secondInput')
        return jsonify({'message': 'Success', 'a': a, 'b': b})
    return render_template('index.html')


@app.route('/button_1/')
def button_1():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'One'})


@app.route('/button_2/')
def button_2():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Two'})


@app.route('/button_3/')
def button_3():
    time.sleep(randrange(1, 2))
    # create a list of image paths inside the static folder
    # For Example: if you have an image name test.png inside static/media/ folder you will omit the static part
    # and the path will be /media/test.png
    # similarly if you have many images inside static/media/ folder you can list them using os.listdir()
    # and then store it in the images variable
    # Note: The media folder was only used as an example, you can use any folder inside static folder
    images = ['/media/test.png', '/media/test.png', '/media/test.png']
    return jsonify({'message': 'Three', 'images': images, 'status': 200}), 200


@app.route('/button_4/')
def button_4():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Four', 'status': 400}), 400


@app.route('/button_5/')
def button_5():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Five'})


@app.route('/button_6/')
def button_6():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Six'})


@app.route('/button_7/')
def button_7():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Seven'})


@app.route('/button_8/')
def button_8():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Eight'})


@app.route('/button_9/')
def button_9():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Nine'})


@app.route('/button_10/')
def button_10():
    time.sleep(randrange(1, 2))
    return jsonify({'message': 'Ten'})


if __name__ == '__main__':
    app.run()
