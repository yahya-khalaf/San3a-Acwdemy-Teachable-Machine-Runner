# app.py
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
    QComboBox,
    QMessageBox,
    QTextEdit,
    QStatusBar,
    QSplitter
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont, QDoubleValidator

import numpy as np
import cv2 # Import cv2 for camera listing

# import core modules
from core.model_wrapper import ModelWrapper
from core.camera_thread import CameraThread
from core.audio_thread import AudioRecorder
from core.image_worker import ImageInferenceWorker
from core.audio_worker import AudioInferenceWorker
from core.serial_worker import SerialWorker
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QPushButton, QLabel
from core.auth import CSVAuth

APP_WIDTH = 900
APP_HEIGHT = 720
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480


class LoginDialog(QDialog):
    def __init__(self, auth: CSVAuth):
        super().__init__()
        self.auth = auth
        self.setWindowTitle("ML Runner Login")

        layout = QVBoxLayout(self)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Username")
        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("Password")
        self.pass_input.setEchoMode(QLineEdit.Password)

        layout.addWidget(QLabel("Username"))
        layout.addWidget(self.user_input)

        layout.addWidget(QLabel("Password"))
        layout.addWidget(self.pass_input)

        # Login button
        btn = QPushButton("Login")
        btn.clicked.connect(self.try_login)
        layout.addWidget(btn)

        # Registration button → opens Google Form
        reg_btn = QPushButton("Register (Open Form)")
        reg_btn.clicked.connect(self.open_register)
        layout.addWidget(reg_btn)

        self.msg = QLabel("")
        self.msg.setStyleSheet("color: red")
        layout.addWidget(self.msg)

    def try_login(self):
        user = self.user_input.text()
        pwd = self.pass_input.text()

        self.msg.setText("Connecting...")
        self.msg.setStyleSheet("color: blue")
        QApplication.processEvents() # Force UI update

        # Updated to handle tuple return (success, message)
        success, message = self.auth.check_login(user, pwd)

        if success:
            self.accept()
        else:
            self.msg.setStyleSheet("color: red")
            self.msg.setText(message)

    def open_register(self):
        import webbrowser
        webbrowser.open("https://forms.gle/hmvS6q5Dq2i3K4iF7")

class TeachableMachineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_wrapper = ModelWrapper()
        self.camera_thread = None
        self.audio_recorder = None
        self.serial_worker = SerialWorker()
        self.image_worker = None
        self.audio_worker = None

        self.confidence_threshold = 0.8
        self.prediction_history = []
        self.prediction_history_size = 6

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        self.setWindowTitle("Teachable Machine Controller")
        self.setGeometry(100, 100, APP_WIDTH, APP_HEIGHT)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = self.create_control_panel()
        right = self.create_preview_panel()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([400, 500])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Idle - No model loaded")

        self.create_menu()

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_action = QAction("Load Model", self)
        load_action.triggered.connect(self.load_model)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        model_box = QGroupBox("Model")
        mlay = QVBoxLayout(model_box)
        self.model_path_label = QLabel("No model loaded")
        mlay.addWidget(self.model_path_label)
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        mlay.addWidget(self.load_btn)
        self.model_info_label = QLabel("Model Type: Unknown")
        mlay.addWidget(self.model_info_label)
        layout.addWidget(model_box)

        # --- Camera Selection Group Box ---
        camera_box = QGroupBox("Camera")
        cl = QVBoxLayout(camera_box)
        self.camera_combo = QComboBox()
        self.refresh_camera_btn = QPushButton("Refresh Cameras")
        self.refresh_camera_btn.clicked.connect(self.refresh_cameras)
        cl.addWidget(self.camera_combo)
        cl.addWidget(self.refresh_camera_btn)
        layout.addWidget(camera_box)
        # -----------------------------------

        serial_box = QGroupBox("Serial")
        sl = QVBoxLayout(serial_box)
        h = QHBoxLayout() if False else None
        self.port_combo = QComboBox()
        self.refresh_ports_btn = QPushButton("Refresh")
        self.refresh_ports_btn.clicked.connect(self.refresh_ports)
        sl.addWidget(self.port_combo)
        sl.addWidget(self.refresh_ports_btn)
        conn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_serial)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_btn.setEnabled(False)
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.disconnect_btn)
        sl.addLayout(conn_layout)
        self.serial_status_label = QLabel("Disconnected")
        sl.addWidget(self.serial_status_label)
        layout.addWidget(serial_box)

        mapping_box = QGroupBox("Label Mapping")
        ml = QVBoxLayout(mapping_box)
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(2)
        self.mapping_table.setHorizontalHeaderLabels(["Label", "Serial Command"])
        ml.addWidget(self.mapping_table)
        layout.addWidget(mapping_box)

        infer_box = QGroupBox("Inference")
        il = QVBoxLayout(infer_box)
        th_layout = QHBoxLayout()
        th_layout.addWidget(QLabel("Confidence Threshold:"))
        self.threshold_input = QLineEdit()
        validator = QDoubleValidator(0.0, 1.0, 2)
        try:
            validator.setNotation(QDoubleValidator.StandardNotation)
        except Exception:
            pass
        self.threshold_input.setValidator(validator)
        self.threshold_input.setText("0.80")
        self.threshold_input.editingFinished.connect(self.on_threshold)
        th_layout.addWidget(self.threshold_input)
        self.threshold_label = QLabel("0.80")
        th_layout.addWidget(self.threshold_label)
        il.addLayout(th_layout)
        ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Inference")
        self.start_btn.clicked.connect(self.start_inference)
        self.stop_btn = QPushButton("Stop Inference")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        il.addLayout(ctrl_layout)
        layout.addWidget(infer_box)

        layout.addStretch()
        return panel

    def create_preview_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        preview_box = QGroupBox("Live Preview")
        pl = QVBoxLayout(preview_box)
        self.preview_label = QLabel("No preview available")
        self.preview_label.setMinimumSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.preview_label)
        layout.addWidget(preview_box)

        preds_box = QGroupBox("Predictions")
        prl = QVBoxLayout(preds_box)
        self.current_prediction_label = QLabel("No predictions yet")
        self.current_prediction_label.setFont(QFont("Arial", 12, QFont.Bold))
        prl.addWidget(self.current_prediction_label)
        self.prediction_details = QTextEdit()
        self.prediction_details.setReadOnly(True)
        self.prediction_details.setMaximumHeight(150)
        prl.addWidget(self.prediction_details)
        layout.addWidget(preds_box)

        status_box = QGroupBox("Status")
        sl = QVBoxLayout(status_box)
        self.status_indicators = {}
        for name in ["Model", "Camera", "Audio", "Serial", "Inference"]:
            lbl = QLabel(f"{name}: Not active")
            self.status_indicators[name] = lbl
            sl.addWidget(lbl)
        layout.addWidget(status_box)
        return panel

    def setup_connections(self):
        self.serial_worker.connection_changed.connect(self.on_serial_connection_changed)
        self.serial_worker.error_occurred.connect(self.on_serial_error)
        self.serial_worker.data_received.connect(self.on_serial_data)
        self.refresh_ports()
        self.refresh_cameras() # Initialize camera list

    def refresh_cameras(self):
        self.camera_combo.clear()
        # Look for available cameras by checking indices 0 through 9
        max_cameras = 10
        for i in range(max_cameras):
            try:
                # Attempt to open the camera with common backends
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Use DSHOW for faster check on Windows
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2) # Try V4L2 for Linux
                    if not cap.isOpened():
                         cap.release()
                         cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) # Try AVFoundation for Mac
                         if not cap.isOpened():
                             cap.release()
                             cap = cv2.VideoCapture(i) # Final generic fallback
                             if not cap.isOpened():
                                 cap.release()
                                 continue
                
                # Try to get a frame to confirm it's a valid device
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.camera_combo.addItem(f"Camera {i}", i) # Store index as data
            except Exception:
                pass 

        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("Default Camera (Index 0)", 0)


    def refresh_ports(self):
        import serial.tools.list_ports
        self.port_combo.clear()
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
        except Exception as e:
            print(f"[app] refresh ports error: {e}")

    def load_model(self):
        # Prevent loading if running (UI lock should handle this, but double check)
        if self.camera_thread or self.audio_recorder:
            QMessageBox.warning(self, "Warning", "Please stop inference before loading a new model.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "TensorFlow Lite (*.tflite)")
        if not file_path:
            return
        labels_guess = file_path.replace(".tflite", "_labels.txt")
        if not os.path.exists(labels_guess):
            labels_guess = os.path.join(os.path.dirname(file_path), "labels.txt")
            if not os.path.exists(labels_guess):
                labels_guess = None
        ok = self.model_wrapper.load_model(file_path, labels_guess)
        if ok:
            self.model_path_label.setText(f"Model: {Path(file_path).name}")
            self.model_info_label.setText(f"Model Type: {self.model_wrapper.model_info.model_type}")
            self.status_indicators["Model"].setText("Model: Loaded")
            self.populate_label_mapping()
            self.update_status("Ready - Model loaded")
        else:
            QMessageBox.critical(self, "Error", "Failed to load model. Ensure it is a valid TFLite file.")

    def populate_label_mapping(self):
        if not self.model_wrapper.model_info:
            return
        labels = self.model_wrapper.model_info.labels
        self.mapping_table.setRowCount(len(labels))
        for i, label in enumerate(labels):
            li = QTableWidgetItem(label)
            li.setFlags(li.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(i, 0, li)
            cmd = label[0].upper() if label else "X"
            ci = QTableWidgetItem(cmd)
            self.mapping_table.setItem(i, 1, ci)

    def toggle_serial(self):
        if not self.serial_worker.connected:
            dev = self.port_combo.currentData()
            if dev:
                self.serial_worker.connect_port(dev, 115200)
                # UI updates handled in on_serial_connection_changed
                self.serial_status_label.setText("Connecting...")
                self.connect_btn.setEnabled(False) # Prevent double click
        else:
            self.disconnect_serial()

    def disconnect_serial(self):
        self.serial_worker.disconnect_port()
        self.serial_status_label.setText("Disconnecting...")
        self.disconnect_btn.setEnabled(False) # Prevent double click

    def on_serial_connection_changed(self, connected: bool):
        # Updates UI based on connection state
        if connected:
            self.serial_status_label.setText("Connected")
            self.status_indicators["Serial"].setText("Serial: Connected")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            
            # LOCK Serial UI controls to prevent changing port while connected
            self.port_combo.setEnabled(False)
            self.refresh_ports_btn.setEnabled(False)
        else:
            self.serial_status_label.setText("Disconnected")
            self.status_indicators["Serial"].setText("Serial: Disconnected")
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            
            # UNLOCK Serial UI controls
            self.port_combo.setEnabled(True)
            self.refresh_ports_btn.setEnabled(True)

    def on_serial_error(self, msg):
        QMessageBox.warning(self, "Serial Error", msg)

    def on_serial_data(self, data):
        print(f"[serial] {data}")

    def on_threshold(self):
        txt = self.threshold_input.text().strip()
        if not txt:
            self.threshold_input.setText("0.80")
            self.confidence_threshold = 0.8
            self.threshold_label.setText("0.80")
            return
        try:
            v = float(txt)
            v = max(0.0, min(1.0, v))
            self.confidence_threshold = v
            self.threshold_label.setText(f"{v:.2f}")
            self.threshold_input.setText(f"{v:.2f}")
        except Exception:
            self.threshold_input.setText(f"{self.confidence_threshold:.2f}")

    def set_inference_ui_locked(self, locked: bool):
        """Locks/Unlocks configuration UI elements during inference."""
        self.load_btn.setEnabled(not locked)
        self.camera_combo.setEnabled(not locked)
        self.refresh_camera_btn.setEnabled(not locked)
        
        self.start_btn.setEnabled(not locked)
        self.stop_btn.setEnabled(locked)

    def start_inference(self):
        if not self.model_wrapper.model_info:
            QMessageBox.warning(self, "Error", "No model loaded")
            return
        
        t = self.model_wrapper.model_info.model_type
        success = False
        if t == "image":
            success = self.start_image_inference()
        elif t == "audio":
            success = self.start_audio_inference()
        else:
            QMessageBox.warning(self, "Error", "Unknown model type")
            
        if success:
            self.set_inference_ui_locked(True)
            self.status_indicators["Inference"].setText("Inference: Running")

    def start_image_inference(self):
        camera_index = self.camera_combo.currentData() if self.camera_combo.currentData() is not None else 0

        self.camera_thread = CameraThread(camera_index=camera_index)
        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.error_occurred.connect(self.on_camera_error)
        
        self.image_worker = ImageInferenceWorker(self.model_wrapper, max_fps=8)
        self.image_worker.prediction_ready.connect(self.on_prediction)
        
        self.camera_thread.start()
        self.image_worker.start()
        
        self.status_indicators["Camera"].setText(f"Camera: Active (Index {camera_index})")
        return True

    def start_audio_inference(self):
        sample_rate = getattr(self.model_wrapper.model_info, "metadata", {}).get("audio", {}).get("sampleRate", 16000)
        
        self.audio_recorder = AudioRecorder(sample_rate=int(sample_rate), chunk_size=1024)
        self.audio_recorder.audio_ready.connect(self.on_audio_chunk)
        self.audio_recorder.error_occurred.connect(self.on_audio_error)
        
        self.audio_worker = AudioInferenceWorker(self.model_wrapper, frames_to_accumulate=4)
        self.audio_worker.prediction_ready.connect(self.on_prediction)
        
        self.audio_recorder.start()
        self.audio_worker.start()
        
        self.status_indicators["Audio"].setText("Audio: Active")
        return True

    def stop_inference(self):
        # stop camera
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            self.status_indicators["Camera"].setText("Camera: Not active")
        if self.image_worker:
            self.image_worker.stop()
            self.image_worker = None

        # stop audio
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder = None
            self.status_indicators["Audio"].setText("Audio: Not active")
        if self.audio_worker:
            self.audio_worker.stop()
            self.audio_worker = None

        self.set_inference_ui_locked(False)
        self.status_indicators["Inference"].setText("Inference: Stopped")
        self.preview_label.setText("No preview available")

    def on_frame(self, frame):
        self.show_preview(frame)
        if self.image_worker:
            self.image_worker.add_frame(frame)

    def on_audio_chunk(self, pcm):
        try:
            peak = int(np.max(np.abs(pcm)))
            self.preview_label.setText(f"Audio Level: {peak}")
        except Exception:
            pass
        if self.audio_worker:
            self.audio_worker.add_audio(pcm)

    def show_preview(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(PREVIEW_WIDTH, PREVIEW_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
        except Exception:
            pass

    def on_prediction(self, preds, label, confidence):
        # smoothing
        self.prediction_history.append(preds)
        if len(self.prediction_history) > self.prediction_history_size:
            self.prediction_history.pop(0)
        avg = np.mean(self.prediction_history, axis=0)
        avg_conf = float(np.max(avg))
        best_idx = int(np.argmax(avg))
        best_label = self.model_wrapper.model_info.labels[best_idx]
        self.current_prediction_label.setText(f"{best_label} ({avg_conf:.2f})")
        details = "\n".join([f"{lab}: {p:.3f}" for lab, p in zip(self.model_wrapper.model_info.labels, avg)])
        self.prediction_details.setText(details)
        # send serial command
        if avg_conf >= self.confidence_threshold and self.serial_worker.connected:
            cmd = self.get_serial_command(best_label)
            if cmd:
                self.serial_worker.send_command(cmd)

    def get_serial_command(self, label):
        for i in range(self.mapping_table.rowCount()):
            item = self.mapping_table.item(i, 0)
            cmd = self.mapping_table.item(i, 1)
            if item and cmd and item.text() == label:
                return cmd.text()
        return None

    def on_camera_error(self, msg):
        QMessageBox.warning(self, "Camera Error", msg)
        self.stop_inference()

    def on_audio_error(self, msg):
        QMessageBox.warning(self, "Audio Error", msg)
        self.stop_inference()

    def update_status(self, text):
        self.status_bar.showMessage(text)

    def show_about(self):
        QMessageBox.about(self, "About", "Teachable Machine Controller - Refactored & Fixed")

    def closeEvent(self, ev):
        # ensure clean shutdown
        try:
            self.stop_inference()
            if self.serial_worker:
                self.serial_worker.stop()
        except Exception:
            pass
        ev.accept()

def main():
    import PySide6.QtGui as QtGui
    QtGui.QGuiApplication.setDesktopFileName("San3a-ML-Runner")

    app = QApplication(sys.argv)

    # your published Google Sheet CSV link:
    CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRr2uwdDTSNhhzzOlfdzJNypggdDuy9ehSscQqGvomS82DHi3iYV1paw-22vjFfVgsCY6faGYoBWUr6/pub?gid=0&single=true&output=csv"

    auth = CSVAuth(CSV_URL)
    login = LoginDialog(auth)

    if login.exec() != QDialog.Accepted:
        return  # Quit the application

    window = TeachableMachineApp()
    window.show()
    window.raise_()
    window.activateWindow()


    sys.exit(app.exec())


if __name__ == "__main__":
    main()