#ifndef DROB_H
#define DROB_H

#include <iostream>
#include <string>

class Drob {
private:
    int* chislitel;   // Chislitel (dinamicheskaya pamyat')
    int* znamenatel; // Znamenatel (dinamicheskaya pamyat')
    std::string* strokovoePredstavlenie; // Strokovoe predstavlenie (dinamicheskaya pamyat')
    static int kolvoDrobey; // Staticheskoe pole: schetchik sozdannykh drobey

public:
    // Konstruktory
    Drob();
    Drob(int chislitel, int znamenatel);
    Drob(const Drob& other); // Konstruktor kopirovaniya

    // Destruktor
    ~Drob();

    // Metody dostupa
    int getChislitel() const;
    int getZnamenatel() const;
    std::string getStrokovoePredstavlenie() const;

    // Metody dlya raboty s drobyami
    Drob slozhenie(const Drob& other) const;
    Drob vichitanie(const Drob& other) const;
    Drob umnozhenie(const Drob& other) const;
    Drob delenie(const Drob& other) const;

    // Staticheskiy metod
    static int getKolvoDrobey();

    // Vyvod drobi
    void print() const;

    // Privedenie k stroke
    void obnovitStrokovoePredstavlenie();
};

#endif
