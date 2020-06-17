from flask import Flask, render_template
from flask_restful import Resource, Api, fields, marshal_with, reqparse
import threading
from real_time_bot_recognition import RealTimeRecognition
import subprocess

app = Flask(__name__)
api = Api(app)


thread = None

parser = reqparse.RequestParser()
parser.add_argument('url', type=str, help='URL for camera')

class StartAnlysis(Resource):
    def get(self):
        return {'it': 'works'}
    
    def post(self, **kwargs):
        global thread
        args = parser.parse_args()
        thread = subprocess.Popen(["python", "real_time_bot_recognition.py", args['url']])
        return render_template('start.html') 
    
class StopAnalysis(Resource):
    def get(self):
        global thread
        # thread.join(1)
        thread.terminate()
        return render_template('stop.html')


api.add_resource(StartAnlysis, '/start', '/health')
api.add_resource(StopAnalysis, '/stop')

if __name__ == '__main__':
    app.run(port=8080, debug=False)