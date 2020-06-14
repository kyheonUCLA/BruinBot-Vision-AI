# Vision and Voice Software for BruinBot

The Vision and Behavior system consists of two subsystems that work concurrently: vision and voice. The information from these two subsystems provide valuable information to make informed actions and reactions. All of these programs are combined and run from a single program that will be run as a server with an accessible API. Since the programs are computation heavy, the server will be run inside a Virtual Machine Instance inside Google Cloud Platform and be called by a program running on the Raspberry Pi on the BruinBot. The subsystems, server, and overall architecture are described below.


## Vision:

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

## Voice:

The voice subsystem uses the attached microphone and speaker to interact with the people. It uses Google Assistant API and Google Actions API to create a conversational user interface that can be run with a command to interact with a user.

