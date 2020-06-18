from flask import Flask, render_template, Response
from flask_restful import Resource, Api, fields, marshal_with, reqparse
import threading
from real_time_bot_recognition import RealTimeRecognition
from subprocess import Popen, PIPE
import time

app = Flask(__name__)
api = Api(app)


thread = None

parser = reqparse.RequestParser()
parser.add_argument('url', type=str, help='URL for camera')

class StartAnlysis(Resource):
    def get(self):
        global thread
        if thread == None:
            return {'not':'running'}
        def inner():
            for line in iter(thread.stdout.readline,''):
                time.sleep(1)                           # Don't need this just shows the text streaming
                yield line.rstrip() + '<br/>\n'

        return Response(inner(), mimetype='text/html')  # text/html is required for most browsers to show th$
    
    def post(self, **kwargs):
        global thread
        args = parser.parse_args()
        thread = Popen(["python", "real_time_bot_recognition.py", args['url']], stdout=PIPE)
        return render_template('start.html') 
    
class StopAnalysis(Resource):
    def get(self):
        global thread
        # thread.join(1)
        thread.terminate()
        return render_template('stop.html')

class Health(Resource):
    def get(self):
        return {'it':'works'}

api.add_resource(StartAnlysis, '/start', '/get_info')
api.add_resource(StopAnalysis, '/stop')
api.add_resource(Health, '/health')

if __name__ == '__main__':
    app.run(port=8080, debug=False)