#!/usr/bin/env python3
"""Сервер UNI-NET, вариант 92, усложнение У3 (checksum)."""

import socket
import struct
import time

PORT = 40644
SIGNATURE = 0xF52C
VARIANT_ID = 1092
VERSION = 1
MAX_PAYLOAD = 4096

HEADER_FORMAT = ">HBBHHH"   # + Checksum (2 байта)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)   # = 10

MSG_HELLO = 0x01
MSG_DATA = 0x02
MSG_STATUS = 0x03
MSG_ERROR = 0x04
MSG_BYE = 0x05


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg))


def checksum(payload: bytes) -> int:
    """Простая сумма байтов по модулю 65536."""
    return sum(payload) & 0xFFFF


def recv_exactly(sock, count):
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def pack_message(msg_type, payload=b""):
    cs = checksum(payload)
    header = struct.pack(HEADER_FORMAT, SIGNATURE, VERSION, msg_type,
                         VARIANT_ID, len(payload), cs)
    return header + payload


def handle_client(conn, addr):
    log("подключение от %s:%d" % addr)
    while True:
        header = recv_exactly(conn, HEADER_SIZE)
        if header is None:
            log("клиент закрыл соединение")
            return

        (signature, version, msg_type, variant_id,
         length, cs) = struct.unpack(HEADER_FORMAT, header)

        log("заголовок: sig=0x%04X ver=%d type=0x%02X var=%d len=%d cs=0x%04X"
            % (signature, version, msg_type, variant_id, length, cs))

        if signature != SIGNATURE:
            log("ОШИБКА: чужая сигнатура")
            conn.sendall(pack_message(MSG_ERROR, b"bad signature"))
            if 0 < length <= MAX_PAYLOAD:
                recv_exactly(conn, length)
            continue

        if length > MAX_PAYLOAD:
            log("ОШИБКА: длина %d больше лимита" % length)
            conn.sendall(pack_message(MSG_ERROR, b"payload too long"))
            continue

        payload = recv_exactly(conn, length) if length else b""
        if payload is None:
            return

        # Проверка контрольной суммы
        actual = checksum(payload)
        if actual != cs:
            log("ОШИБКА: checksum не совпал (пришло 0x%04X, посчитано 0x%04X)"
                % (cs, actual))
            conn.sendall(pack_message(MSG_ERROR, b"bad checksum"))
            continue

        if msg_type == MSG_HELLO:
            log("HELLO payload=%r" % payload)
            conn.sendall(pack_message(MSG_STATUS, b"hello accepted"))
        elif msg_type == MSG_DATA:
            log("DATA payload=%r" % payload)
            conn.sendall(pack_message(MSG_STATUS, b"ok:" + payload[:32]))
        elif msg_type == MSG_STATUS:
            log("STATUS")
            conn.sendall(pack_message(MSG_STATUS, b"server alive"))
        elif msg_type == MSG_BYE:
            log("BYE")
            conn.sendall(pack_message(MSG_BYE))
            return
        else:
            log("ОШИБКА: неизвестный тип 0x%02X" % msg_type)
            conn.sendall(pack_message(MSG_ERROR, b"unknown type"))


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", PORT))
        srv.listen(5)
        log("сервер UNI-NET слушает порт %d (вариант 92, У3)" % PORT)
        while True:
            conn, addr = srv.accept()
            with conn:
                try:
                    handle_client(conn, addr)
                except Exception as e:
                    log("исключение: %s" % e)


if __name__ == "__main__":
    main()