#include "drob.h"
#include <numeric> // Dlya std::gcd

int Drob::kolvoDrobey = 0;

//----------------------Konstruktory----------------------
Drob::Drob() : chislitel(new int(0)), znamenatel(new int(1)), strokovoePredstavlenie(new std::string("0/1")) {
    kolvoDrobey++;
}

Drob::Drob(int chislitel, int znamenatel) : chislitel(new int(chislitel)), znamenatel(new int(znamenatel)), strokovoePredstavlenie(new std::string("")) {
    if (znamenatel == 0) {
        std::cerr << "Oshibka: Znamenatel ne mozhet byt' nulem. Ustanovka v 1." << std::endl;
        *this->znamenatel = 1;
    }
    obnovitStrokovoePredstavlenie();
    kolvoDrobey++;
}

Drob::Drob(const Drob& other) : chislitel(new int(*other.chislitel)), znamenatel(new int(*other.znamenatel)), strokovoePredstavlenie(new std::string(*other.strokovoePredstavlenie)) {
    kolvoDrobey++;
}
//----------------------Konstruktory----------------------

Drob::~Drob() {
    delete chislitel;
    delete znamenatel;
    delete strokovoePredstavlenie;
    kolvoDrobey--;
}

int Drob::getChislitel() const {
    return *chislitel;
}

int Drob::getZnamenatel() const {
    return *znamenatel;
}

std::string Drob::getStrokovoePredstavlenie() const {
    return *strokovoePredstavlenie;
}

int Drob::getKolvoDrobey() {
    return kolvoDrobey;
}

void Drob::obnovitStrokovoePredstavlenie() {
    int obshiyDelitel = std::gcd(*chislitel, *znamenatel);
    int uproschenniyChislitel = *chislitel / obshiyDelitel;
    int uproschenniyZnamenatel = *znamenatel / obshiyDelitel;

    // Obnovlyaem stroku, vydelyaya novuyu pamyat' pod novoe znachenie
    delete strokovoePredstavlenie;
    strokovoePredstavlenie = new std::string(std::to_string(uproschenniyChislitel) + "/" + std::to_string(uproschenniyZnamenatel));
}

Drob Drob::slozhenie(const Drob& other) const {
    int noviyChislitel = (*chislitel * *other.znamenatel) + (*other.chislitel * *znamenatel);
    int noviyZnamenatel = *znamenatel * *other.znamenatel;
    return Drob(noviyChislitel, noviyZnamenatel);
}

Drob Drob::vichitanie(const Drob& other) const {
    int noviyChislitel = (*chislitel * *other.znamenatel) - (*other.chislitel * *znamenatel);
    int noviyZnamenatel = *znamenatel * *other.znamenatel;
    return Drob(noviyChislitel, noviyZnamenatel);
}

Drob Drob::umnozhenie(const Drob& other) const {
    int noviyChislitel = *chislitel * *other.chislitel;
    int noviyZnamenatel = *znamenatel * *other.znamenatel;
    return Drob(noviyChislitel, noviyZnamenatel);
}

Drob Drob::delenie(const Drob& other) const {
    if (*other.chislitel == 0) {
        std::cerr << "Oshibka: Delenie na nol'." << std::endl;
        return *this;
    }
    int noviyChislitel = *chislitel * *other.znamenatel;
    int noviyZnamenatel = *znamenatel * *other.chislitel;
    return Drob(noviyChislitel, noviyZnamenatel);
}

void Drob::print() const {
    std::cout << *strokovoePredstavlenie << std::endl;
}
