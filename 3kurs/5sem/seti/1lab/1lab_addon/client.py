#!/usr/bin/env python3
"""Клиент UNI-NET, вариант 92, усложнение У3 (checksum)."""

import socket
import struct
import time

HOST = "127.0.0.1"
PORT = 40644
SIGNATURE = 0xF52C
VARIANT_ID = 1092
VERSION = 1

HEADER_FORMAT = ">HBBHHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

MSG_HELLO = 0x01
MSG_DATA = 0x02
MSG_STATUS = 0x03
MSG_ERROR = 0x04
MSG_BYE = 0x05


def checksum(payload: bytes) -> int:
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


def pack_message(msg_type, payload=b"", signature=SIGNATURE,
                 length=None, cs=None):
    if length is None:
        length = len(payload)
    if cs is None:
        cs = checksum(payload)
    header = struct.pack(HEADER_FORMAT, signature, VERSION, msg_type,
                         VARIANT_ID, length, cs)
    return header + payload


def read_reply(sock):
    header = recv_exactly(sock, HEADER_SIZE)
    if header is None:
        print("сервер закрыл соединение")
        return None
    (signature, version, msg_type, variant_id,
     length, cs) = struct.unpack(HEADER_FORMAT, header)
    payload = recv_exactly(sock, length) if length else b""
    print("ответ: тип=0x%02X, cs=0x%04X, payload=%r"
          % (msg_type, cs, payload))
    return msg_type, payload


def main():
    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        sock.sendall(pack_message(MSG_HELLO, b"client v1"))
        read_reply(sock)

        # DATA, тип П1 — текстовые измерения ключ=значение
        samples = [
            b"temp=23.5",
            b"pressure=101.3",
            b"humidity=45",
        ]
        for s in samples:
            sock.sendall(pack_message(MSG_DATA, s))
            read_reply(sock)
            time.sleep(0.1)

        # Некорректное #1: чужая сигнатура
        sock.sendall(pack_message(MSG_DATA, b"sig=bad", signature=0x0000))
        read_reply(sock)

        # Некорректное #2: завышенная длина
        sock.sendall(pack_message(MSG_DATA, b"", length=5000, cs=0))
        read_reply(sock)

        # Некорректное #3: битая контрольная сумма (это и есть У3!)
        sock.sendall(pack_message(MSG_DATA, b"temp=99.9", cs=0xDEAD))
        read_reply(sock)

        # STATUS
        sock.sendall(pack_message(MSG_STATUS))
        read_reply(sock)

        # BYE
        sock.sendall(pack_message(MSG_BYE))
        read_reply(sock)

    print("сеанс завершен")


if __name__ == "__main__":
    main()