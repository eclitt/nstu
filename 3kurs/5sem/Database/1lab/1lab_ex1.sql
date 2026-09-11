DROP VIEW IF EXISTS car_report CASCADE;
DROP TABLE IF EXISTS cars CASCADE;
DROP TABLE IF EXISTS car_models CASCADE;
DROP TABLE IF EXISTS car_specs CASCADE;
DROP TABLE IF EXISTS motors CASCADE;
DROP TABLE IF EXISTS proizvoditel CASCADE;
DROP TABLE IF EXISTS countries CASCADE;

DROP TYPE IF EXISTS bar_type CASCADE;
DROP TYPE IF EXISTS weel_type CASCADE;
DROP TYPE IF EXISTS transmission_type CASCADE;
DROP TYPE IF EXISTS fuel_type CASCADE;
DROP TYPE IF EXISTS fuel_add_type CASCADE;
DROP TYPE IF EXISTS car_type CASCADE;

-- 3.1 Справочник стран
CREATE TABLE countries (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

-- 3.2 Производители
CREATE TABLE proizvoditel (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE
);

-- 3.3 Двигатели
CREATE TABLE motors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    fuel VARCHAR(20) NOT NULL,
    fuel_add VARCHAR(20) NOT NULL
);

-- 3.4 Технические характеристики
CREATE TABLE car_specs (
    id SERIAL PRIMARY KEY,
    bar VARCHAR(20) NOT NULL,
    weels VARCHAR(20) NOT NULL,
    transmission VARCHAR(20) NOT NULL,
    motor_id INTEGER REFERENCES motors(id) NOT NULL,
    body_type VARCHAR(20) NOT NULL
);

-- 3.5 Модели автомобилей
CREATE TABLE car_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    spec_id INTEGER REFERENCES car_specs(id) NOT NULL,
    proizvoditel_id INTEGER REFERENCES proizvoditel(id) ON DELETE CASCADE,
    country_id INTEGER REFERENCES countries(id)
);

-- 3.6 ГЛАВНАЯ ТАБЛИЦА — Автомобили
CREATE TABLE cars (
    id SERIAL PRIMARY KEY,
    car_name VARCHAR(100) NOT NULL,
    car_model_id INTEGER REFERENCES car_models(id) ON DELETE CASCADE,
    proizvoditel_id INTEGER REFERENCES proizvoditel(id) ON DELETE CASCADE,
    year_of_manufacture DATE NOT NULL,
    mileage INTEGER NOT NULL DEFAULT 0,
    price DECIMAL(12, 2) NOT NULL,
    color VARCHAR(30)
);

-- 4.1 Страны
INSERT INTO countries (name) VALUES
    ('Япония'),
    ('Германия'),
    ('Россия'),
    ('США'),
    ('Южная Корея');

-- 4.2 Производители
INSERT INTO proizvoditel (name, country_id) VALUES
    ('Toyota', 1),
    ('Honda', 1),
    ('Volkswagen', 2),
    ('АвтоВАЗ', 3),
    ('Ford', 4),
    ('Hyundai', 5);

-- 4.3 Двигатели
INSERT INTO motors (name, fuel, fuel_add) VALUES
    ('1.6 MPI', 'АИ-92', 'Полный'),
    ('2.0 TDI', 'ДТ', 'Полу'),
    ('1.8 VTEC', 'АИ-95', 'Полный'),
    ('2.5 Hybrid', 'АИ-95', 'Полный'),
    ('Электро 150кВт', 'Электро', 'Полный');

-- 4.4 Характеристики
INSERT INTO car_specs (bar, weels, transmission, motor_id, body_type) VALUES
    ('Левый', 'Передний', 'Автомат', 1, 'Седан'),
    ('Правый', 'Полный', 'Автомат', 2, 'Внедорожник'),
    ('Левый', 'Передний', 'Механика', 3, 'Хэтчбек'),
    ('Левый', 'Передний', 'Механика', 1, 'Седан'),
    ('Левый', 'Полный', 'Автомат', 4, 'Кроссовер'),
    ('Левый', 'Передний', 'Автомат', 5, 'Хэтчбек');

-- 4.5 Модели
INSERT INTO car_models (name, spec_id, proizvoditel_id, country_id) VALUES
    ('Camry', 1, 1, 1),
    ('Land Cruiser', 2, 1, 1),
    ('Golf', 3, 3, 2),
    ('Vesta', 4, 4, 3),
    ('CR-V', 5, 2, 1),
    ('Ioniq', 6, 6, 5);

-- 4.6 Автомобили
INSERT INTO cars (car_name, car_model_id, proizvoditel_id, year_of_manufacture, mileage, price, color) VALUES
    ('Toyota Camry XLE', 1, 1, '2020-01-01', 15000, 2500000.00, 'Белый'),
    ('Toyota Land Cruiser V8', 2, 1, '2019-01-01', 45000, 5000000.00, 'Чёрный'),
    ('Volkswagen Golf 7', 3, 3, '2021-01-01', 5000, 2200000.00, 'Серый'),
    ('Lada Vesta SW', 4, 4, '2022-01-01', 0, 1200000.00, 'Синий'),
    ('Honda CR-V', 5, 2, '2020-01-01', 0,3500000.00, 'Красный'),
    ('Hyundai Ioniq', 6, 6, '2021-01-01', 20000, 2800000.00, 'Белый');

CREATE VIEW car_report AS
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
    c.price AS "Цена"
FROM cars c
INNER JOIN proizvoditel p ON c.proizvoditel_id = p.id
INNER JOIN countries cnt ON p.country_id = cnt.id
INNER JOIN car_models cm ON c.car_model_id = cm.id
INNER JOIN car_specs cs ON cm.spec_id = cs.id
INNER JOIN motors m ON cs.motor_id = m.id;

SELECT * FROM car_report;
