#include "drob.h"
#include <iostream>

int main() {
    // Sozdanie drobey
    Drob f1(1, 2);
    Drob f2(1, 3);

    std::cout << "Kolvo drobey: " << Drob::getKolvoDrobey() << std::endl;

    // Vyvod drobey
    std::cout << "f1: ";
    f1.print();
    std::cout << "f2: ";
    f2.print();

    // Arifmeticheskie operatsii
    Drob summa = f1.slozhenie(f2);
    std::cout << "Summa: ";
    summa.print();

    Drob raznost = f1.vichitanie(f2);
    std::cout << "Raznost': ";
    raznost.print();

    Drob proizvedenie = f1.umnozhenie(f2);
    std::cout << "Proizvedenie: ";
    proizvedenie.print();

    Drob chastnoe = f1.delenie(f2);
    std::cout << "Chastnoe: ";
    chastnoe.print();

     std::cout << "Kolvo drobey: " << Drob::getKolvoDrobey() << std::endl;
    return 0;
}
