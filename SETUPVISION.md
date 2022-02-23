## Getting Started with Vision (Linux)

1) Clone github repo to local directory.
2) Create a venv using python's venv module: python3 -m venv venv
3) Activate venv: source venv/bin/activate  Now the console should look like: (venv) $
4) Use pip to install the necessary modules: pip3 install -r requirements.txt
5) Go to http://console.cloud.google.com/ and create a service account key (instructions: https://cloud.google.com/iam/docs/creating-managing-service-account-keys)
6) Make sure that the json key is called "key.json" and put it in the same directory as real_time_bot_recognition.py

# Running Image Recognition:

1) run: python3 imageRecognition.py "filepath/to/image"
2) The output will be put into the 'test_outputs' folder

# Running Real Time Recognition

1) run: python3 realTimeRecognition.py
