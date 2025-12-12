# led_controller.py
"""
import serial
import time

class LEDController:
    def __init__(self, port="COM3", baudrate=115200):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            print("[LED] Serial connected")
        except Exception as e:
            print("[LED] ERROR opening serial:", e)
            self.ser = None

    def send(self, mode: str):
        if self.ser is None:
            print("[LED] Serial not available")
            return

        try:
            msg = mode.strip().upper() + "\n"
            self.ser.write(msg.encode())

            resp = self.ser.readline().decode(errors="ignore").strip()
            print(f"[LED] Sent '{mode}', Received → {resp}")

        except Exception as e:
            print("[LED] ERROR sending command:", e)


# Initialize global controller
led = LEDController(port="COM3", baudrate=115200)
"""

import serial
import time

class LEDController:
    def __init__(self, port="COM8", baud=9600):
        """
        port: COM port where Arduino UNO is connected
        baud: must match Serial.begin() in Arduino (9600)
        """
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)   # wait for Arduino to reboot
            print(f"[LED] Connected to Arduino on {port}")
        except Exception as e:
            print(f"[LED] ERROR opening port {port}: {e}")
            self.ser = None

    def send(self, mode: str):
        if not self.ser:
            print("[LED] Serial not available")
            return

        mode = mode.strip().upper()
        try:
            self.ser.write((mode + "\n").encode())
            print(f"[LED] Sent '{mode}'")
        except Exception as e:
            print(f"[LED] ERROR sending '{mode}': {e}")


# Initialize global controller
# CHANGE COM PORT IF NEEDED (Windows: COM3, COM4, etc. / Linux: /dev/ttyUSB0)
led = LEDController(port="COM8", baud=9600)
