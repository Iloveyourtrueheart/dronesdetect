from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import threading
import queue
import time
import uuid
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
app = Flask(__name__)

# 全局配置
model_weights = "app\\best.pt"  # 修改为您的模型路径
model_conf = 0.4
model_max_det = 100
MAX_USERS = 3

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())  # 唯一会话ID
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.cap = None
        self.last_active = time.time()
        self.model = YOLO(model_weights).to('cuda')  # 每个会话独立模型实例
        self.model.conf = model_conf
        self.model.max_det = model_max_det
        self.cap = None

    def start_stream(self, stream_url):
        # 使用更完整的 RTSP 参数
        gst_str = (
            f"rtspsrc location={stream_url} latency=0 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            # 尝试备用方法
            self.cap = cv2.VideoCapture(stream_url)
            if not self.cap.isOpened():
                return False
        return True    
    
    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_running = False

user_sessions = {}
session_lock = threading.Lock()

def capture_thread(session):
    while session.is_running:
        try:
            ret, frame = session.cap.read()
            if not ret:
                print(f"无法从流中读取帧，会话 {session.session_id}")
                break
            if session.frame_queue.qsize() < 10:
                resized_frame = cv2.resize(frame, (640, 360))
                session.frame_queue.put(resized_frame)
            time.sleep(0.01)
        except Exception as e:
            print(f"捕获线程错误: {str(e)}")
            break


def cleanup_inactive_sessions():
    while True:
        time.sleep(10)
        with session_lock:
            inactive_users = [uid for uid, session in user_sessions.items() 
                            if time.time() - session.last_active > 30]
            for uid in inactive_users:
                user_sessions[uid].cleanup()
                del user_sessions[uid]

threading.Thread(target=cleanup_inactive_sessions, daemon=True).start()

def capture_thread(session):
    while session.is_running:
        ret, frame = session.cap.read()
        if not ret:
            break
        if session.frame_queue.qsize() < 10:
            resized_frame = cv2.resize(frame, (640, 360))
            session.frame_queue.put(resized_frame)
        time.sleep(0.01)

def generate_frames(session):
    while session.is_running:
        if not session.frame_queue.empty():
            frame = session.frame_queue.get()
            results = session.model(frame, half=True, batch=1,classes=[1])  # 使用会话独立的模型
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            session.last_active = time.time()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/start', methods=['POST'])
def start_detection():
    user_id = request.remote_addr
    stream_url = request.form['stream_url']
    
    with session_lock:
        if len(user_sessions) >= MAX_USERS:
            return jsonify({
                "status": "error",
                "message": f"已达到最大用户数({MAX_USERS})，请稍后再试"
            })
        
        if user_id in user_sessions:
            user_sessions[user_id].cleanup()
        
        session = UserSession(user_id)
        if not session.start_stream(stream_url):  # 使用新的连接方法
            return jsonify({"status": "error", "message": "无法打开视频流！"})
            
        user_sessions[user_id] = session
        session.is_running = True
        threading.Thread(target=capture_thread, args=(session,), daemon=True).start()
        
        return jsonify({
            "status": "success",
            "session_id": session.session_id,
            "current_users": len(user_sessions),
            "max_users": MAX_USERS
        })
    

@app.route('/stop')
def stop_detection():
    user_id = request.remote_addr
    with session_lock:
        if user_id in user_sessions:
            user_sessions[user_id].cleanup()
            del user_sessions[user_id]
    return jsonify({
        "status": "stopped",
        "current_users": len(user_sessions)
    })

@app.route('/video_feed')
def video_feed():
    user_id = request.remote_addr
    session_id = request.args.get('session_id')
    
    if user_id not in user_sessions or not user_sessions[user_id].is_running:
        return "Session not found", 404
    
    if session_id and user_sessions[user_id].session_id != session_id:
        return "Session ID mismatch", 403
        
    return Response(generate_frames(user_sessions[user_id]),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/user_count')
def get_user_count():
    return jsonify({
        "current_users": len(user_sessions),
        "max_users": MAX_USERS
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)