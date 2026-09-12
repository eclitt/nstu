#!/usr/bin/env python3
"""Стартовый каркас сервера UNI-NET. Заполните места, помеченные TODO.

Параметры своего варианта возьмите из таблицы variants.csv:
порт, сигнатура (signature) и идентификатор варианта (variant_id).
"""

import socket
import struct

# TODO: подставьте параметры своего варианта
PORT = 0            # порт из варианта
SIGNATURE = 0x0000  # сигнатура из варианта
VARIANT_ID = 0      # идентификатор варианта
VERSION = 1

HEADER_FORMAT = ">HBBHH"   # Signature, Version, Type, VariantID, Length
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

MSG_HELLO = 0x01
MSG_DATA = 0x02
MSG_STATUS = 0x03
MSG_ERROR = 0x04
MSG_BYE = 0x05


def recv_exactly(sock, count):
    """Читает из сокета ровно count байт или возвращает None при EOF.

    Вспомните лекцию 5: TCP не сохраняет границы сообщений, поэтому
    читать нужно в цикле до получения нужного числа байтов.
    """
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


def handle_client(conn, addr):
    print("подключение от", addr)
    while True:
        header = recv_exactly(conn, HEADER_SIZE)
        if header is None:
            print("клиент закрыл соединение")
            return
        signature, version, msg_type, variant_id, length = struct.unpack(
            HEADER_FORMAT, header
        )

        # TODO: проверьте сигнатуру; при несовпадении отправьте ERROR
        # TODO: ограничьте максимальную длину payload

        payload = recv_exactly(conn, length) if length else b""
        if payload is None:
            return

        # TODO: запишите событие в журнал (файл или print с меткой времени)

        if msg_type == MSG_HELLO:
            conn.sendall(pack_message(MSG_STATUS, b"hello accepted"))
        elif msg_type == MSG_DATA:
            # TODO: сформируйте осмысленный ответ STATUS
            conn.sendall(pack_message(MSG_STATUS, b"ok"))
        elif msg_type == MSG_BYE:
            conn.sendall(pack_message(MSG_BYE))
            return
        else:
            # TODO: отправьте ERROR для неизвестного типа сообщения
            pass


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", PORT))
        srv.listen(5)
        print("сервер UNI-NET слушает порт", PORT)
        while True:
            conn, addr = srv.accept()
            with conn:
                handle_client(conn, addr)


if __name__ == "__main__":
    main()
