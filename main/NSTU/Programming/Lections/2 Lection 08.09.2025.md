# Константы
```
DEFINE INT_MAX = 99999 // Пример константы в с
const int ptr = 999999 // Пример константы в C++
```
___
# Inline функции

В inline функции не преобразуются:
- Рукурсивные
- Цикдические
- switch \ goto
- Функции с переменным количеством аргументов

```
inline T foo (int args) {/* function code*/}
```
___
# Оператор
**Оператор** - особый конструкт языка, ведущий себя как функция.
___
# Перегрузка оператора
```
T operator operator_type (U &obj)
```
Где:
- operator_type - тип оператора
- Т - тип возвращаемого значения,
- U - тип аргумента
В зависимости от типа оператора может быть ноль, один, два или больше аргументов.

## Дружественная функция

```
class Int {
friend Int operator-(const Int& left, const Int &right) ;
} ;
Int operator - (const Int& left, const Int &right) /
return Int (left. value - right.value) ;
```
## Глобальная функция
```
bool operator == (const Int& left, const Int &right) {
	return left.getValue() == right.getValue()
	}
```
## Что можно перегружать?
 - =
 -  ->
 -  ()
 - []
 - ->*

# Оператор присваивания
```
T operator = (const T &op);
Foo a;
Fpp b = a;
a = b;
```
## Что нельзя перегружать
- Доступ к полям (.)
- Разыменовывание (*)
- ...
- ...
- ...

