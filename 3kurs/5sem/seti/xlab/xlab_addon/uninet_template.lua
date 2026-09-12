-- uninet_template.lua: стартовый каркас Lua-диссектора UNI-NET.
-- Заполните места, помеченные TODO, параметрами своего варианта
-- и недостающими полями.
--
-- Установка: скопируйте файл в каталог плагинов Wireshark
-- (Help > About Wireshark > Folders > Personal Lua Plugins)
-- или запустите: wireshark -X lua_script:uninet_template.lua

-- TODO: подставьте параметры своего варианта
local MY_PORT = 0            -- порт из варианта
local MY_SIGNATURE = 0x0000  -- сигнатура из варианта

local uninet = Proto("uninet", "UNI-NET Protocol")

-- Поля протокола: каждое поле становится доступным в display filter
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
-- TODO: добавьте поля uninet.variant (uint16) и uninet.length (uint16)
-- TODO: добавьте поле uninet.payload (bytes)

uninet.fields = { f_signature, f_version, f_type }

local HEADER_SIZE = 8

function uninet.dissector(buffer, pinfo, tree)
    if buffer:len() < HEADER_SIZE then
        -- Заголовок пришел не целиком: просим TCP доставить остаток.
        -- Подумайте: почему одного пакета может быть недостаточно?
        pinfo.desegment_len = DESEGMENT_ONE_MORE_SEGMENT
        return
    end

    pinfo.cols.protocol = "UNI-NET"

    local subtree = tree:add(uninet, buffer(), "UNI-NET Message")
    subtree:add(f_signature, buffer(0, 2))
    subtree:add(f_version, buffer(2, 1))
    subtree:add(f_type, buffer(3, 1))
    -- TODO: разберите поля Variant ID (смещение 4) и Payload Length
    -- (смещение 6)
    -- TODO: отобразите Payload (начиная со смещения 8, длина из заголовка)
    -- TODO: обработайте случай, когда сообщение разрезано на несколько
    -- TCP-сегментов (pinfo.desegment_offset и pinfo.desegment_len)
    -- TODO (усложнение): в одном сегменте может быть несколько сообщений
end

local tcp_table = DissectorTable.get("tcp.port")
tcp_table:add(MY_PORT, uninet)
-- Для UDP-вариантов используйте DissectorTable.get("udp.port")
