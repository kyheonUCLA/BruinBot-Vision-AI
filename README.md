# Vision and Voice Software for BruinBot

The Vision and Behavior system consists of two subsystems that work concurrently: vision and voice. The information from these two subsystems provide valuable information to make informed actions and reactions. All of these programs are combined and run from a single program that will be run as a server with an accessible API. Since the programs are computation heavy, the server will be run inside a Virtual Machine Instance inside Google Cloud Platform and be called by a program running on the Raspberry Pi on the BruinBot. The subsystems, server, and overall architecture are described below.

# Overall

## Cloud for Vision

1. On camera, go to ./streaming and run `python stream.py`
    - This will create a webserver that sends camera feed to localhost:5000
    - Use app like ngrok to broadcast localhost:5000 to a pulic url
        - Very simple to do: https://dashboard.ngrok.com/get-started/setup
2. Next, ssh into the vm and install all requirements with `pip install -r requirements.txt`
3. You need to download json credentials from google cloud console and add path location in real_time_bot_recognition.py
4. Then, in the vm run `sudo gunicorn api:app -b 0.0.0.0:8080` to run the api server
5. This will be accessible via EXTERNAL_API_ADDRESS:8000
    - For now that is http://35.222.123.238:8080/

## API
- `POST /start`
    - Needs one parameters
    - {URL: camera url ("...ngrok.io/video_feed") from step 1}
    - Starts analysis using the camera and saves data to firestore
- `GET /start`
    - Run only after POST
    - Gets the data from real time analysis
- `GET /stop`
    - Stops real time analysis

## Audio
### Run using the given instructions in the voice folder

Link to the google actions project to set up locally
https://console.actions.google.com/u/0/project/bruinbot-443ea/overview

## Individual

### Vision:

The vision subsystem uses the attached camera to detect humans and their emotions. The script takes in live video feed and detects humans face and legs; once it detects a face, it will try to detect emotions. The software utilizes Tensorflow (Keras API) and OpenCV with Python. This information will allow the BruinBot to know if it should approach a person.

### Instructions:
#### Real Time Video
* If not on GCP Compute Engine, install Google Cloud SDK locally
* Clone github repo
* Add in JSON Credentials in real_time_bot_recognition.py
* Install necessary packages (tensorflow, numpy, opencv, google.cloud)
* Run: python real_time_bot_recognition.py

#### Image Recognition
* If not on GCP Compute Engine, install Google Cloud SDK
* Clone github repo
* Add in JSON Credentials in image_bot_recognition.py
* Install necessary packages (tensorflow, numpy, opencv, google.cloud)
* Run: python image_bot_recognition.py "*path to image*"

#### Reading Output
* Script will output to both Terminal and Firebase in following format:
* For Face: face,*emotion*,*height of box*,*length of box*,timestamp
* For Legs: legs,*height of box*,*length of box*,timestamp
* If image is processed, outgoing image will be stored in /test_output

### Voice:

The voice subsystem uses the attached microphone and speaker to interact with the people. It uses Google Assistant API and Google Actions API to create a conversational user interface that can be run with a command to interact with a user.

