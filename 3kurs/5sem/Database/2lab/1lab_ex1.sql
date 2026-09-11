CREATE USER autosalon_user WITH PASSWORD 'autosalon123';
GRANT ALL PRIVILEGES ON DATABASE autosalon_db TO autosalon_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO autosalon_user;
ALTER USER autosalon_user WITH CREATEDB;
ALTER USER autosalon_user WITH SUPERUSER;

\connect autosalon_db autosalon_user localhost

-- 1. Ограничение на привод (в таблице car_specs)
ALTER TABLE car_specs
ADD CONSTRAINT check_weels
CHECK (weels IN ('Передний', 'Задний', 'Полный'));

-- 2. Ограничение на руль
ALTER TABLE car_specs
ADD CONSTRAINT check_bar
CHECK (bar IN ('Левый', 'Правый'));

-- 3. Ограничение на тип кузова
ALTER TABLE car_specs
ADD CONSTRAINT check_body_type
CHECK (body_type IN ('Седан', 'Универсал', 'Минивэн', 'Хэтчбек', 'Кроссовер', 'Внедорожник'));

-- 4. Ограничение на дату выпуска (не больше текущей)
ALTER TABLE cars
ADD CONSTRAINT check_year
CHECK (year_of_manufacture <= EXTRACT(YEAR FROM CURRENT_DATE));

insert into cars values (7,'Totyota ne toyota',1,1,2028,15000, 2000,'Белый', 'XXX', 'Новый','Продан', 'Картой');
insert into cars values (7,'Totyota ne toyota',1,1,2025,15000, 2000,'Белый', 'XXX', 'Новый','Продан', 'Картой');

insert into car_specs values (7, 'Средний', 'Никакой', 'Автомат',1, 'Седан');
insert into car_specs values (7, 'Левый', 'Никакой', 'Автомат', 1, 'Седан');
insert into car_specs values (7, 'Левый', 'Полный', 'Автомат', 1, 'Седан');
