import os
import base64
import imghdr
import tempfile
from nsfw_detector import predict
from flask import Flask, request, jsonify

if not os.path.exists('./tmp'):
    os.makedirs('./tmp')

app = Flask(__name__)
model = predict.load_model('./models/mobilenet_v2_140_224/saved_model.h5')


@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        image_data = base64.b64decode(image_base64)

        # Check the image extension based on image data
        image_extension = imghdr.what(None, h=image_data)
        if not image_extension:
            return jsonify({'error': 'Invalid or unsupported image format'}), 400

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=image_extension, delete=False, dir='./tmp') as temp_file:
            temp_filename = temp_file.name
            temp_file.write(image_data)

        # Process the image data
        result = predict.classify(model, temp_filename)
        first_key = next(iter(result))
        first_value = result[first_key]

        os.remove(temp_filename)
        return jsonify(first_value)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=False)
