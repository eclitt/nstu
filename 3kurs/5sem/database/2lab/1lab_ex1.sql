CREATE USER autosalon_user WITH PASSWORD 'autosalon123';
GRANT ALL PRIVILEGES ON DATABASE autosalon_db TO autosalon_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO autosalon_user;
ALTER USER autosalon_user WITH CREATEDB;
ALTER USER autosalon_user WITH SUPERUSER;

\connect autosalon_db autosalon_user localhost

ALTER TABLE car_specs
ADD CONSTRAINT check_weels
CHECK (weels IN ('Передний', 'Задний', 'Полный'));

ALTER TABLE car_specs
ADD CONSTRAINT check_bar
CHECK (bar IN ('Левый', 'Правый'));

ALTER TABLE car_specs
ADD CONSTRAINT check_body_type
CHECK (body_type IN ('Седан', 'Универсал', 'Минивэн', 'Хэтчбек', 'Кроссовер', 'Внедорожник'));

-- 4. Ограничение на дату выпуска (не больше текущей)
ALTER TABLE cars
ADD CONSTRAINT check_year
CHECK (year_of_manufacture <= CURRENT_DATE);insert into cars values (7,'Totyota ne toyota',1,1,2028,15000, 2000,'Белый', 'XXX', 'Новый','Продан', 'Картой');
insert into cars values (7,'Totyota ne toyota',1,1,2025,15000, 2000,'Белый', 'XXX', 'Новый','Продан', 'Картой');

INSERT INTO car_specs VALUES (7, 'Средний', 'Передний', 'Автомат', 1, 'Седан');
INSERT INTO car_specs VALUES (7, 'Левый', 'Передний', 'Автомат', 1, 'Седан');

INSERT INTO car_specs VALUES (8, 'Левый', 'ХЗ', 'Автомат', 1, 'Седан');
INSERT INTO car_specs VALUES (8, 'Левый', 'Передний', 'Автомат', 1, 'Седан');

INSERT INTO car_specs VALUES (9, 'Левый', 'Передний', 'Автомат', 1, '123');
INSERT INTO car_specs VALUES (9, 'Левый', 'Передний', 'Автомат', 1, 'Седан');

BEGIN;

-- Добавляем произвольные поля (например, в cars)
ALTER TABLE cars ADD COLUMN test_field TEXT;
ALTER TABLE cars ADD COLUMN another_field INTEGER;

-- Создаём производную таблицу (VIEW)
CREATE VIEW test_view AS
SELECT car_name, price FROM cars WHERE price > 2000000;

-- Просматриваем структуру
\d cars
\d test_view

INSERT INTO cars (car_name, car_model_id, proizvoditel_id, year_of_manufacture, price)
VALUES ('Parallel Test', 1, 1, 2024, 999999);

ROLLBACK;
