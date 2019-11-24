import base64

import googleapiclient.discovery
import numpy as np

from scipy.io import wavfile

fs, data = wavfile.read('test.wav')
print(data)
print(data.min(), data.max())

enc = base64.b64encode(open("test.wav", "rb").read()).decode()

PROJECT_ID = "gifted-honor-259919"
MODEL_NAME = 'birdDetectionModel'
VERSION_NAME = 'v1'

new_data = np.frombuffer(base64.b64decode(enc), dtype=np.int16)
print(new_data)
print(new_data.min(), new_data.max())
wavfile.write('test1.wav', fs, new_data)

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}/versions/{}'.format(PROJECT_ID, MODEL_NAME, VERSION_NAME)

response = service.projects().predict(
    name=name,
    body={'instances': enc}
).execute()

if 'error' in response:
    print(response)
    raise RuntimeError(response['error'])
else:
    print(response['predictions'])
