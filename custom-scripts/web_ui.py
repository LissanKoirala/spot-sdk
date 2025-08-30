import json
import os
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
from threading import Thread

# --- Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
RESULTS_TOPIC = "spot/results"
WEB_ASSETS_DIR = "web_assets"

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='gevent')

# --- MQTT Client Setup ---
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Web UI connected to MQTT Broker!")
        client.subscribe(RESULTS_TOPIC)
        print(f"Subscribed to topic: {RESULTS_TOPIC}")
    else:
        print(f"Web UI failed to connect to MQTT, return code {rc}\n")

def on_message(client, userdata, msg):
    """Callback for when a new result message is received."""
    print(f"Received result from topic {msg.topic}")
    try:
        result_data = json.loads(msg.payload.decode())
        # Emit the data to all connected web clients
        socketio.emit('new_result', result_data, namespace='/')
        print("Emitted result to web clients.")
    except json.JSONDecodeError:
        print("Error decoding JSON from result message.")
    except Exception as e:
        print(f"An error occurred in on_message: {e}")


def mqtt_listener():
    """Runs the MQTT client loop."""
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"Could not start MQTT listener thread: {e}")


# --- Flask Routes ---
@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets (processed images)."""
    return send_from_directory(WEB_ASSETS_DIR, filename)


if __name__ == '__main__':
    # Start the MQTT listener in a background thread
    mqtt_thread = Thread(target=mqtt_listener, daemon=True)
    mqtt_thread.start()
    
    print("Starting Flask-SocketIO server...")
    # Use gevent server for SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)