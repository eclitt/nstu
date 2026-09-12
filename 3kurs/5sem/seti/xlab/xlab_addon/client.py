#!/usr/bin/env python3
"""Стартовый каркас клиента UNI-NET. Заполните места, помеченные TODO."""

import socket
import struct

# TODO: подставьте параметры своего варианта
HOST = "127.0.0.1"
PORT = 0            # порт из варианта
SIGNATURE = 0x0000  # сигнатура из варианта
VARIANT_ID = 0      # идентификатор варианта
VERSION = 1

HEADER_FORMAT = ">HBBHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

MSG_HELLO = 0x01
MSG_DATA = 0x02
MSG_STATUS = 0x03
MSG_ERROR = 0x04
MSG_BYE = 0x05


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
    header = struct.pack(HEADER_FORMAT, SIGNATURE, VERSION, msg_type,
                         VARIANT_ID, len(payload))
    return header + payload


def read_reply(sock):
    header = recv_exactly(sock, HEADER_SIZE)
    if header is None:
        print("сервер закрыл соединение")
        return None
    signature, version, msg_type, variant_id, length = struct.unpack(
        HEADER_FORMAT, header
    )
    payload = recv_exactly(sock, length) if length else b""
    print("ответ: тип=0x%02X, payload=%r" % (msg_type, payload))
    return msg_type, payload


def main():
    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        sock.sendall(pack_message(MSG_HELLO, b"client v1"))
        read_reply(sock)

        # TODO: отправьте несколько сообщений DATA с полезной нагрузкой,
        # соответствующей типу данных вашего варианта (payload_kind)

        # TODO: отправьте одно некорректное сообщение
        # (неверная сигнатура, завышенная длина или неизвестный тип)
        # и покажите реакцию сервера

        sock.sendall(pack_message(MSG_BYE))
        read_reply(sock)
    print("сеанс завершен")


if __name__ == "__main__":
    main()
