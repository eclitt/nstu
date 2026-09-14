-- uninet.lua: диссектор UNI-NET, вариант 92, У3 (checksum)

local MY_PORT = 40644
local MY_SIGNATURE = 0xF52C

local uninet = Proto("uninet", "UNI-NET Protocol")

local f_signature = ProtoField.uint16("uninet.signature", "Signature",
    base.HEX)
local f_version = ProtoField.uint8("uninet.version", "Version", base.DEC)
local f_type = ProtoField.uint8("uninet.type", "Message Type", base.HEX, {
    [0x01] = "HELLO",
    [0x02] = "DATA",
    [0x03] = "STATUS",
    [0x04] = "ERROR",
    [0x05] = "BYE",
})
local f_variant = ProtoField.uint16("uninet.variant", "Variant ID", base.DEC)
local f_length  = ProtoField.uint16("uninet.length", "Payload Length",
    base.DEC)
local f_checksum = ProtoField.uint16("uninet.checksum", "Checksum", base.HEX)
local f_payload = ProtoField.bytes("uninet.payload", "Payload")

uninet.fields = { f_signature, f_version, f_type, f_variant, f_length,
                  f_checksum, f_payload }

local HEADER_SIZE = 10   -- было 8, добавили 2 байта checksum

function uninet.dissector(buffer, pinfo, tree)
    local total = buffer:len()
    local offset = 0

    while offset < total do
        if total - offset < HEADER_SIZE then
            pinfo.desegment_offset = offset
            pinfo.desegment_len = DESEGMENT_ONE_MORE_SEGMENT
            return
        end

        local payload_len = buffer(offset + 6, 2):uint()
        local msg_total = HEADER_SIZE + payload_len

        if total - offset < msg_total then
            pinfo.desegment_offset = offset
            pinfo.desegment_len = msg_total - (total - offset)
            return
        end

        pinfo.cols.protocol = "UNI-NET"

        local msg_type = buffer(offset + 3, 1):uint()
        local variant  = buffer(offset + 4, 2):uint()

        local subtree = tree:add(uninet, buffer(offset, msg_total),
                                 "UNI-NET Message")
        subtree:add(f_signature, buffer(offset, 2))
        subtree:add(f_version,   buffer(offset + 2, 1))
        subtree:add(f_type,      buffer(offset + 3, 1))
        subtree:add(f_variant,   buffer(offset + 4, 2))
        subtree:add(f_length,    buffer(offset + 6, 2))
        subtree:add(f_checksum,  buffer(offset + 8, 2))

        if payload_len > 0 then
            local payload_tvb = buffer(offset + 10, payload_len)
            subtree:add(f_payload, payload_tvb)

            -- Проверяем контрольную сумму
            local declared = buffer(offset + 8, 2):uint()
            local bytes = payload_tvb:bytes()
            local actual = 0
            for i = 0, bytes:len() - 1 do
                actual = (actual + bytes:get_index(i)) % 65536
            end
            if actual ~= declared then
                subtree:add_expert_info(PI_CHECKSUM, PI_WARN,
                    string.format("Checksum не совпал: 0x%04X != 0x%04X",
                                  declared, actual))
            end
        end

        local type_name = ({
            [0x01] = "HELLO", [0x02] = "DATA", [0x03] = "STATUS",
            [0x04] = "ERROR", [0x05] = "BYE",
        })[msg_type] or "UNKNOWN"

        pinfo.cols.info:append(string.format(
            "UNI-NET %s var=%d len=%d", type_name, variant, payload_len))

        offset = offset + msg_total
    end
end

local tcp_table = DissectorTable.get("tcp.port")
tcp_table:add(MY_PORT, uninet)