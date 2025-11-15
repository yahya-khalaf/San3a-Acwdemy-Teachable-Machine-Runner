# core/serial_worker.py
from PySide6.QtCore import QThread, Signal
import serial
import serial.tools.list_ports
import queue
import time

class SerialWorker(QThread):
    connection_changed = Signal(bool)
    error_occurred = Signal(str)
    data_received = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._port = None
        self._baud = 115200
        self._serial = None
        self._command_q = queue.Queue()
        self._connect_request = False
        self._disconnect_request = False
        self.connected = False

    def connect_port(self, port: str, baudrate: int = 115200):
        self._port = port
        self._baud = baudrate
        self._connect_request = True
        if not self.isRunning():
            self.start()

    def disconnect_port(self):
        self._disconnect_request = True

    def send_command(self, cmd: str):
        if cmd:
            self._command_q.put(cmd)

    def run(self):
        self._running = True
        while self._running:
            try:
                if self._connect_request and not self.connected:
                    try:
                        if self._serial and self._serial.is_open:
                            self._serial.close()
                        self._serial = serial.Serial(self._port, self._baud, timeout=1.0)
                        self.connected = True
                        self.connection_changed.emit(True)
                    except Exception as e:
                        self.error_occurred.emit(f"Serial connect error: {e}")
                        self.connected = False
                    self._connect_request = False

                if self._disconnect_request and self.connected:
                    try:
                        if self._serial and self._serial.is_open:
                            self._serial.close()
                        self.connected = False
                        self.connection_changed.emit(False)
                    except Exception as e:
                        self.error_occurred.emit(f"Serial disconnect error: {e}")
                    self._disconnect_request = False

                if self.connected and self._serial and self._serial.is_open:
                    # send commands
                    while not self._command_q.empty():
                        cmd = self._command_q.get()
                        try:
                            self._serial.write(cmd.encode('utf-8'))
                            self._serial.flush()
                        except Exception as e:
                            self.error_occurred.emit(f"Serial write error: {e}")
                    # non-blocking read
                    try:
                        if self._serial.in_waiting:
                            line = self._serial.readline().decode('utf-8', errors='ignore').strip()
                            if line:
                                self.data_received.emit(line)
                    except Exception:
                        pass
            except Exception as e:
                self.error_occurred.emit(f"Serial loop error: {e}")
            time.sleep(0.01)
        # cleanup
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass
        self.connected = False
        self.connection_changed.emit(False)

    def stop(self):
        self._running = False
        self.wait()
