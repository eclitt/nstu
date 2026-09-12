
-- 1. VIN-номер
ALTER TABLE cars ADD COLUMN vin_number VARCHAR(17) UNIQUE;

UPDATE cars SET vin_number = CASE id
    WHEN 1 THEN 'JTNBE46K123456789'
    WHEN 2 THEN 'JTMHV05J204123456'
    WHEN 3 THEN 'WVWZZZ1KZAW123456'
    WHEN 4 THEN 'XTAGFK330AB123456'
    WHEN 5 THEN 'JHMFA16586S123456'
    WHEN 6 THEN 'KMHC851GACU123456'
END;

-- 2. Состояние авто
CREATE TYPE car_condition AS ENUM ('Новый', 'Б/У', 'Битый', 'На запчасти');
ALTER TABLE cars ADD COLUMN "condition" car_condition;

UPDATE cars SET "condition" = CASE id
    WHEN 1 THEN 'Новый'::car_condition
    WHEN 2 THEN 'Б/У'::car_condition
    WHEN 3 THEN 'Б/У'::car_condition
    WHEN 4 THEN 'Новый'::car_condition
    WHEN 5 THEN 'Новый'::car_condition
    WHEN 6 THEN 'Б/У'::car_condition
END;

-- 3. Статус продажи
CREATE TYPE sale_status AS ENUM ('В наличии', 'Продан', 'Забронирован', 'Снят с продажи');
ALTER TABLE cars ADD COLUMN status sale_status;

UPDATE cars SET status = CASE id
    WHEN 1 THEN 'В наличии'::sale_status
    WHEN 2 THEN 'Продан'::sale_status
    WHEN 3 THEN 'Забронирован'::sale_status
    WHEN 4 THEN 'В наличии'::sale_status
    WHEN 5 THEN 'В наличии'::sale_status
    WHEN 6 THEN 'Снят с продажи'::sale_status
END;

-- 4. Способ оплаты
CREATE TYPE payment_method AS ENUM ('Наличные', 'Картой', 'Банковский перевод', 'Кредит', 'Рассрочка');
ALTER TABLE cars ADD COLUMN payment_method payment_method;

UPDATE cars SET payment_method = CASE id
    WHEN 1 THEN 'Картой'::payment_method
    WHEN 2 THEN 'Наличные'::payment_method
    WHEN 3 THEN 'Кредит'::payment_method
    WHEN 4 THEN 'Рассрочка'::payment_method
    WHEN 5 THEN 'Картой'::payment_method
    WHEN 6 THEN 'Банковский перевод'::payment_method
END;

-- 5. Проверка таблицы cars
SELECT id, car_name, vin_number, "condition", status, payment_method, price FROM cars;

-- 6. Обновляем VIEW car_report
CREATE OR REPLACE VIEW car_report1 AS
SELECT
    c.id AS "ID",
    c.car_name AS "Название авто",
    p.name AS "Производитель",
    cnt.name AS "Страна",
    cm.name AS "Модель",
    cs.body_type AS "Кузов",
    cs.bar AS "Руль",
    cs.weels AS "Привод",
    cs.transmission AS "Коробка",
    m.name AS "Двигатель",
    m.fuel AS "Топливо",
    c.year_of_manufacture AS "Год",
    c.mileage AS "Пробег",
    c.color AS "Цвет",
    c.vin_number AS "VIN",
    c."condition" AS "Состояние",
    c.status AS "Статус",
    c.payment_method AS "Оплата",
    c.price AS "Цена"
FROM cars c
INNER JOIN proizvoditel p ON c.proizvoditel_id = p.id
INNER JOIN countries cnt ON p.country_id = cnt.id
INNER JOIN car_models cm ON c.car_model_id = cm.id
INNER JOIN car_specs cs ON cm.spec_id = cs.id
INNER JOIN motors m ON cs.motor_id = m.id;

-- 7. Финальная проверка отчёта
SELECT * FROM car_report;
