#!/usr/bin/env python3
"""Runtime helper for DH PGC gripper control over Modbus RTU."""

import time

import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu


class DHPGCGripper:
    def __init__(self, port="/dev/ttyUSB0", slave_id=1, baudrate=115200, timeout=0.5, retries=3, retry_wait_s=0.08):
        self.port = port
        self.slave_id = int(slave_id)
        self.retries = max(1, int(retries))
        self.retry_wait_s = max(0.0, float(retry_wait_s))
        self.master = modbus_rtu.RtuMaster(
            serial.Serial(
                port=port,
                baudrate=int(baudrate),
                bytesize=8,
                parity="N",
                stopbits=1,
                xonxoff=False,
            )
        )
        self.master.set_timeout(float(timeout))

    def _reset_serial_buffers(self):
        serial_port = getattr(self.master, "_serial", None)
        if serial_port is None:
            return
        for method_name in ("reset_input_buffer", "reset_output_buffer"):
            method = getattr(serial_port, method_name, None)
            if method is not None:
                method()

    def _execute(self, *args, **kwargs):
        last_error = None
        for attempt in range(self.retries):
            try:
                return self.master.execute(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                self._reset_serial_buffers()
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_wait_s)
        raise last_error

    def init_gripper(self, full_calibration=False, timeout=5.0):
        value = 0xA5 if full_calibration else 0x01
        self._execute(self.slave_id, cst.WRITE_SINGLE_REGISTER, 0x0100, output_value=value)
        time.sleep(3.0 if full_calibration else 1.0)
        start = time.time()
        while time.time() - start < float(timeout):
            if self.get_init_status() == 1:
                return True
            time.sleep(0.2)
        return False

    def set_force(self, force_percent):
        force_percent = int(force_percent)
        if force_percent < 20 or force_percent > 100:
            raise ValueError("force must be in [20, 100]")
        self._execute(self.slave_id, cst.WRITE_SINGLE_REGISTER, 0x0101, output_value=force_percent)

    def set_speed(self, speed_percent):
        speed_percent = int(speed_percent)
        if speed_percent < 1 or speed_percent > 100:
            raise ValueError("speed must be in [1, 100]")
        self._execute(self.slave_id, cst.WRITE_SINGLE_REGISTER, 0x0104, output_value=speed_percent)

    def set_position(self, position):
        position = int(position)
        if position < 0 or position > 1000:
            raise ValueError("position must be in [0, 1000]")
        self._execute(self.slave_id, cst.WRITE_SINGLE_REGISTER, 0x0103, output_value=position)

    def get_init_status(self):
        result = self._execute(self.slave_id, cst.READ_HOLDING_REGISTERS, 0x0200, 1)
        return int(result[0])

    def get_grip_status(self):
        result = self._execute(self.slave_id, cst.READ_HOLDING_REGISTERS, 0x0201, 1)
        return int(result[0])

    def get_position(self):
        result = self._execute(self.slave_id, cst.READ_HOLDING_REGISTERS, 0x0202, 1)
        return int(result[0])

    def wait_until_done(self, timeout=3.0):
        start = time.time()
        last_status = None
        while time.time() - start < float(timeout):
            last_status = self.get_grip_status()
            if last_status in (1, 2, 3):
                return last_status
            time.sleep(0.1)
        return last_status

    def close(self):
        self.master.close()
