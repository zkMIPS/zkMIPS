#pragma once

#include <cstdio>
#include "prelude.hpp"
#include "kb31_t.hpp"

#ifdef __CUDA_ARCH__
#define FUN __host__ __device__
#endif
#ifndef __CUDA_ARCH__
#define FUN inline
#endif

class kb31_cipolla_t {
    public:
        kb31_t real;
        kb31_t imag;

        FUN kb31_cipolla_t(kb31_t real, kb31_t imag) {
            this->real = kb31_t(real);
            this->imag = kb31_t(imag);
        }

        FUN static kb31_cipolla_t one() {
            return kb31_cipolla_t(kb31_t::one(), kb31_t::zero());
        }

        FUN kb31_cipolla_t mul_ext(kb31_cipolla_t other, kb31_t nonresidue) {
            kb31_t new_real = real * other.real + nonresidue * imag * other.imag;
            kb31_t new_imag = real * other.imag + imag * other.real;
            return kb31_cipolla_t(new_real, new_imag);
        }

        FUN kb31_cipolla_t pow(uint32_t exponent, kb31_t nonresidue) {
            kb31_cipolla_t result = kb31_cipolla_t::one();
            kb31_cipolla_t base = *this;

            while(exponent) {
                if(exponent & 1) {
                    result = result.mul_ext(base, nonresidue);
                }
                exponent >>= 1;
                base = base.mul_ext(base, nonresidue);
            }

            return result;
        }
};

namespace constants {
    #ifdef __CUDA_ARCH__
        __constant__ constexpr const kb31_t frobenius_const[49] = {
            kb31_t(int(1)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)),
            kb31_t(int(587483156)), kb31_t(int(843070426)), kb31_t(int(856916903)), kb31_t(int(802055410)), kb31_t(int(1274370027)), kb31_t(int(839777993)), kb31_t(int(1763169463)),
            kb31_t(int(1211185764)), kb31_t(int(536911287)), kb31_t(int(1786731555)), kb31_t(int(1891857573)), kb31_t(int(591969516)), kb31_t(int(550155966)), kb31_t(int(706525029)),
            kb31_t(int(926148950)), kb31_t(int(97341948)), kb31_t(int(1328592391)), kb31_t(int(2024338901)), kb31_t(int(1053611575)), kb31_t(int(858809194)), kb31_t(int(895371293)),
            kb31_t(int(1525385643)), kb31_t(int(1541060576)), kb31_t(int(1544460289)),  kb31_t(int(1695665723)), kb31_t(int(1260084848)), kb31_t(int(209013872)), kb31_t(int(1422484900)),
            kb31_t(int(636881039)), kb31_t(int(1369380874)), kb31_t(int(1823056783)), kb31_t(int(411001166)), kb31_t(int(474370133)), kb31_t(int(1991878855)), kb31_t(int(193955070)),
            kb31_t(int(448462982)), kb31_t(int(1809047550)), kb31_t(int(1873051132)), kb31_t(int(1563342685)), kb31_t(int(638206204)), kb31_t(int(1034022669)), kb31_t(int(616721146))
        };

        __constant__ constexpr const kb31_t double_frobenius_const[49] = {
            kb31_t(int(1)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)),
            kb31_t(int(850855402)), kb31_t(int(83752463)), kb31_t(int(578907183)), kb31_t(int(1077461187)), kb31_t(int(841195559)), kb31_t(int(707516819)), kb31_t(int(141214579)),
            kb31_t(int(836146895)), kb31_t(int(2043859405)), kb31_t(int(2072756292)), kb31_t(int(685210173)), kb31_t(int(510761813)), kb31_t(int(193547797)), kb31_t(int(310193486)),
            kb31_t(int(1605797233)), kb31_t(int(989471584)), kb31_t(int(1210699680)), kb31_t(int(1003960530)), kb31_t(int(1444517609)), kb31_t(int(759580625)), kb31_t(int(1114273922)),
            kb31_t(int(1181931158)), kb31_t(int(511865135)), kb31_t(int(172170608)), kb31_t(int(1549372938)), kb31_t(int(153489079)), kb31_t(int(1246252776)), kb31_t(int(1044577581)),
            kb31_t(int(682248311)), kb31_t(int(1022876955)), kb31_t(int(1873346400)), kb31_t(int(850875418)), kb31_t(int(605656029)), kb31_t(int(190509635)), kb31_t(int(220419312)),
            kb31_t(int(688846502)), kb31_t(int(1836380477)), kb31_t(int(172054673)), kb31_t(int(688169080)), kb31_t(int(187745906)), kb31_t(int(414105003)), kb31_t(int(756944866))
        };

        __constant__ constexpr const kb31_t dummy_x[7] = {kb31_t(int(1706420302)), kb31_t(int(1319108093)), kb31_t(int(148224806)), kb31_t(int(26874985)), kb31_t(int(1766171812)), kb31_t(int(1645633948)), kb31_t(int(2028659224))};
        __constant__ constexpr const kb31_t dummy_y[7] = {kb31_t(int(942390502)), kb31_t(int(1239997438)), kb31_t(int(458866455)), kb31_t(int(1843332012)), kb31_t(int(1309764648)), kb31_t(int(572807436)), kb31_t(int(74267719))};

        __constant__ constexpr kb31_t start_x[7] = {kb31_t(int(637514027)), kb31_t(int(1595065213)), kb31_t(int(1998064738)), kb31_t(int(72333738)), kb31_t(int(1211544370)), kb31_t(int(822986770)), kb31_t(int(1518535784))};
        __constant__ constexpr kb31_t start_y[7] = {kb31_t(int(1604177449)), kb31_t(int(90440090)), kb31_t(int(259343427)), kb31_t(int(140470264)), kb31_t(int(1162099742)), kb31_t(int(941559812)), kb31_t(int(1064053343))};

    #endif

    #ifndef __CUDA_ARCH__
        static constexpr const kb31_t frobenius_const[49] = {
            kb31_t(int(1)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)),
            kb31_t(int(587483156)), kb31_t(int(843070426)), kb31_t(int(856916903)), kb31_t(int(802055410)), kb31_t(int(1274370027)), kb31_t(int(839777993)), kb31_t(int(1763169463)),
            kb31_t(int(1211185764)), kb31_t(int(536911287)), kb31_t(int(1786731555)), kb31_t(int(1891857573)), kb31_t(int(591969516)), kb31_t(int(550155966)), kb31_t(int(706525029)),
            kb31_t(int(926148950)), kb31_t(int(97341948)), kb31_t(int(1328592391)), kb31_t(int(2024338901)), kb31_t(int(1053611575)), kb31_t(int(858809194)), kb31_t(int(895371293)),
            kb31_t(int(1525385643)), kb31_t(int(1541060576)), kb31_t(int(1544460289)),  kb31_t(int(1695665723)), kb31_t(int(1260084848)), kb31_t(int(209013872)), kb31_t(int(1422484900)),
            kb31_t(int(636881039)), kb31_t(int(1369380874)), kb31_t(int(1823056783)), kb31_t(int(411001166)), kb31_t(int(474370133)), kb31_t(int(1991878855)), kb31_t(int(193955070)),
            kb31_t(int(448462982)), kb31_t(int(1809047550)), kb31_t(int(1873051132)), kb31_t(int(1563342685)), kb31_t(int(638206204)), kb31_t(int(1034022669)), kb31_t(int(616721146))
        };

        static constexpr const kb31_t double_frobenius_const[49] = {
            kb31_t(int(1)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)), kb31_t(int(0)),
            kb31_t(int(850855402)), kb31_t(int(83752463)), kb31_t(int(578907183)), kb31_t(int(1077461187)), kb31_t(int(841195559)), kb31_t(int(707516819)), kb31_t(int(141214579)),
            kb31_t(int(836146895)), kb31_t(int(2043859405)), kb31_t(int(2072756292)), kb31_t(int(685210173)), kb31_t(int(510761813)), kb31_t(int(193547797)), kb31_t(int(310193486)),
            kb31_t(int(1605797233)), kb31_t(int(989471584)), kb31_t(int(1210699680)), kb31_t(int(1003960530)), kb31_t(int(1444517609)), kb31_t(int(759580625)), kb31_t(int(1114273922)),
            kb31_t(int(1181931158)), kb31_t(int(511865135)), kb31_t(int(172170608)), kb31_t(int(1549372938)), kb31_t(int(153489079)), kb31_t(int(1246252776)), kb31_t(int(1044577581)),
            kb31_t(int(682248311)), kb31_t(int(1022876955)), kb31_t(int(1873346400)), kb31_t(int(850875418)), kb31_t(int(605656029)), kb31_t(int(190509635)), kb31_t(int(220419312)),
            kb31_t(int(688846502)), kb31_t(int(1836380477)), kb31_t(int(172054673)), kb31_t(int(688169080)), kb31_t(int(187745906)), kb31_t(int(414105003)), kb31_t(int(756944866))
        };

        static constexpr kb31_t dummy_x[7] = {kb31_t(int(1706420302)), kb31_t(int(1319108093)), kb31_t(int(148224806)), kb31_t(int(26874985)), kb31_t(int(1766171812)), kb31_t(int(1645633948)), kb31_t(int(2028659224))};
        static constexpr kb31_t dummy_y[7] = {kb31_t(int(942390502)), kb31_t(int(1239997438)), kb31_t(int(458866455)), kb31_t(int(1843332012)), kb31_t(int(1309764648)), kb31_t(int(572807436)), kb31_t(int(74267719))};

        static constexpr kb31_t start_x[7] = {kb31_t(int(637514027)), kb31_t(int(1595065213)), kb31_t(int(1998064738)), kb31_t(int(72333738)), kb31_t(int(1211544370)), kb31_t(int(822986770)), kb31_t(int(1518535784))};
        static constexpr kb31_t start_y[7] = {kb31_t(int(1604177449)), kb31_t(int(90440090)), kb31_t(int(259343427)), kb31_t(int(140470264)), kb31_t(int(1162099742)), kb31_t(int(941559812)), kb31_t(int(1064053343))};

    #endif     
}   

class kb31_septic_extension_t {
    // The value of KoalaBear septic extension element.
    public:
        kb31_t value[7];    
        static constexpr const kb31_t* frobenius_const = constants::frobenius_const;
        static constexpr const kb31_t* double_frobenius_const = constants::double_frobenius_const;

        FUN kb31_septic_extension_t() {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                this->value[i] = kb31_t(0);
            }
        } 

        FUN kb31_septic_extension_t(kb31_t value) {
            this->value[0] = value;
            for (uintptr_t i = 1 ; i < 7 ; i++) {
                this->value[i] = kb31_t(0);
            }
        }

        FUN kb31_septic_extension_t(kb31_t value[7]) {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                this->value[i] = value[i];
            }
        }

        FUN kb31_septic_extension_t(const kb31_t value[7]) {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                this->value[i] = value[i];
            }
        }

        static FUN kb31_septic_extension_t zero() {
            return kb31_septic_extension_t();
        }

        static FUN kb31_septic_extension_t one() {
            return kb31_septic_extension_t(kb31_t::one());
        }

        static FUN kb31_septic_extension_t two() {
            return kb31_septic_extension_t(kb31_t::two());
        }

        static FUN kb31_septic_extension_t from_canonical_u32(uint32_t n) {
            return kb31_septic_extension_t(kb31_t::from_canonical_u32(n));
        }

        FUN kb31_septic_extension_t& operator+=(const kb31_t b) {
            value[0] += b;
            return *this;
        }

        friend FUN kb31_septic_extension_t operator+(kb31_septic_extension_t a, const kb31_t b) {
            return a += b;
        }

        FUN kb31_septic_extension_t& operator+=(const kb31_septic_extension_t b) {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                value[i] += b.value[i];
            }
            return *this;
        }

        friend FUN kb31_septic_extension_t operator+(kb31_septic_extension_t a, const kb31_septic_extension_t b) {
            return a += b;
        }

        FUN kb31_septic_extension_t& operator-=(const kb31_t b) {
            value[0] -= b;
            return *this;
        }

        friend FUN kb31_septic_extension_t operator-(kb31_septic_extension_t a, const kb31_t b) {
            return a -= b;
        }

        FUN kb31_septic_extension_t& operator-=(const kb31_septic_extension_t b) {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                value[i] -= b.value[i];
            }
            return *this;
        }

        friend FUN kb31_septic_extension_t operator-(kb31_septic_extension_t a, const kb31_septic_extension_t b) {
            return a -= b;
        }

        FUN kb31_septic_extension_t& operator*=(const kb31_t b) {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                value[i] *= b;
            }
            return *this;
        }

        friend FUN kb31_septic_extension_t operator*(kb31_septic_extension_t a, const kb31_t b) {
            return a *= b;
        }

        FUN kb31_septic_extension_t& operator*=(const kb31_septic_extension_t b) {
            kb31_t res[13] = {};
            for(uintptr_t i = 0 ; i < 13 ; i++) {
                res[i] = kb31_t::zero();
            }
            for(uintptr_t i = 0 ; i < 7 ; i++) {
                for(uintptr_t j = 0 ; j < 7 ; j++) {
                    res[i + j] += value[i] * b.value[j];
                }
            }
            for(uintptr_t i = 7 ; i < 13 ; i++) {
                res[i - 7] += res[i] * kb31_t::from_canonical_u32(8);
                res[i - 6] -= res[i] * kb31_t::from_canonical_u32(2);
            }
            for(uintptr_t i = 0 ; i < 7 ; i++) {
                value[i] = res[i];
            }
            return *this;
        }  

        friend FUN kb31_septic_extension_t operator*(kb31_septic_extension_t a, const kb31_septic_extension_t b) {
            return a *= b;
        }

        FUN bool operator==(const kb31_septic_extension_t rhs) const {
             for(uintptr_t i = 0 ; i < 7 ; i++) {
                if(value[i] != rhs.value[i]) {
                    return false;
                }
            }
            return true;
        }

        FUN kb31_septic_extension_t frobenius() const {
            kb31_t res[7] = {};
            res[0] = value[0];
            for(uintptr_t i = 1 ; i < 7 ; i++) {
                res[i] = kb31_t::zero();
            }
            for(uintptr_t i = 1 ; i < 7 ; i++) {
                for(uintptr_t j = 0 ; j < 7 ; j++) {
                    res[j] += value[i] * frobenius_const[7 * i + j];
                }
            }
            return kb31_septic_extension_t(res);

        }

        FUN kb31_septic_extension_t double_frobenius() const {
            kb31_t res[7] = {};
            res[0] = value[0];
            for(uintptr_t i = 1 ; i < 7 ; i++) {
                res[i] = kb31_t::zero();
            }
            for(uintptr_t i = 1 ; i < 7 ; i++) {
                for(uintptr_t j = 0 ; j < 7 ; j++) {
                    res[j] += value[i] * double_frobenius_const[7 * i + j];
                }
            }
            return kb31_septic_extension_t(res);

        }

        FUN kb31_septic_extension_t pow_r_1() const {
            kb31_septic_extension_t base = frobenius();
            base *= double_frobenius();
            kb31_septic_extension_t base_p2 = base.double_frobenius();
            kb31_septic_extension_t base_p4 = base_p2.double_frobenius();
            return base * base_p2 * base_p4;
        }

        FUN kb31_t pow_r() const {
            kb31_septic_extension_t pow_r1 = pow_r_1();
            kb31_septic_extension_t pow_r = pow_r1 * *this;
            return pow_r.value[0];
        }

        FUN kb31_septic_extension_t reciprocal() const {
            kb31_septic_extension_t pow_r1 = pow_r_1();
            kb31_septic_extension_t pow_r = pow_r1 * *this;
            return pow_r1 * pow_r.value[0].reciprocal();
        }

        friend FUN kb31_septic_extension_t operator/(kb31_septic_extension_t a, kb31_septic_extension_t b) {
            return a * b.reciprocal();
        }

        FUN kb31_septic_extension_t& operator/=(const kb31_septic_extension_t a) {
            return *this *= a.reciprocal();
        }

        FUN kb31_septic_extension_t sqrt(kb31_t pow_r) const {
            if (*this == kb31_septic_extension_t::zero()) {
                return *this;
            }

            kb31_septic_extension_t n_iter = *this;
            kb31_septic_extension_t n_power = *this;
            for(uintptr_t i = 1 ; i < 30 ; i++) {
                n_iter *= n_iter;
                if(i >= 23) {
                    n_power *= n_iter;
                }
            }

            kb31_septic_extension_t n_frobenius = n_power.frobenius();
            kb31_septic_extension_t denominator = n_frobenius;

            n_frobenius = n_frobenius.double_frobenius();
            denominator *= n_frobenius;
            n_frobenius = n_frobenius.double_frobenius();
            denominator *= n_frobenius;
            denominator *= *this;

            kb31_t base = pow_r.reciprocal();
            kb31_t g = kb31_t::from_canonical_u32(3);
            kb31_t a = kb31_t::one();
            kb31_t nonresidue = kb31_t::one() - base;

            while (true) {
                kb31_t is_square = nonresidue ^ 1065353216;
                if (is_square != kb31_t::one()) {
                    break;
                }
                a *= g;
                nonresidue = a.square() - base;
            }

            kb31_cipolla_t x = kb31_cipolla_t(a, kb31_t::one());
            x = x.pow(1065353217, nonresidue);

            return denominator * x.real;
        }

        FUN kb31_septic_extension_t curve_formula() const {
            kb31_septic_extension_t result = *this * *this * *this;
            kb31_t t[7] = { kb31_t(0u), kb31_t(int(3)), kb31_t(0u), kb31_t(0u), kb31_t(0u), kb31_t(0u), kb31_t(0u)};
            result += *this * kb31_septic_extension_t(t);
            result.value[0] -= kb31_t::from_canonical_u32(3);
            return result;
        }

        // Digest sign bands — mirror of `septic_extension.rs::{RECEIVE_Y6_MAX,
        // SEND_Y6_MIN}` and of the `GlobalLookupOperation` AIR.
        static const uint32_t RECEIVE_Y6_MAX = 63u << 24;
        static const uint32_t SEND_Y6_MIN = (1u << 30) + 1u;

        FUN bool is_receive() const {
            uint32_t limb = value[6].as_canonical_u32();
            return 1 <= limb && limb <= RECEIVE_Y6_MAX;
        }

        FUN bool is_send() const {
            uint32_t limb = value[6].as_canonical_u32();
            return SEND_Y6_MIN <= limb && limb <= (kb31_t::MOD - 1);
        }

        FUN bool is_exception() const {
            return !is_receive() && !is_send();
        }
};


class kb31_septic_curve_t {
    public:
        kb31_septic_extension_t x;
        kb31_septic_extension_t y;

        static constexpr const kb31_t* dummy_x = constants::dummy_x;
        static constexpr const kb31_t* dummy_y = constants::dummy_y;
        static constexpr const kb31_t* start_x = constants::start_x;
        static constexpr const kb31_t* start_y = constants::start_y;
        
        FUN kb31_septic_curve_t() {
            this->x = kb31_septic_extension_t::zero();
            this->y = kb31_septic_extension_t::zero();
        }

        FUN kb31_septic_curve_t(kb31_septic_extension_t x, kb31_septic_extension_t y) {
            this->x = x;
            this->y = y;
        }

        FUN kb31_septic_curve_t(kb31_t value[14]) {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                this->x.value[i] = value[i];
            }
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                this->y.value[i] = value[i + 7];
            }
        }

        FUN kb31_septic_curve_t(kb31_t value_x[7], kb31_t value_y[7]) {
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                this->x.value[i] = value_x[i];
                this->y.value[i] = value_y[i];
            }
        }

        static FUN kb31_septic_curve_t dummy_point() {
            kb31_septic_extension_t x;
            kb31_septic_extension_t y;
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                x.value[i] = dummy_x[i];
                y.value[i] = dummy_y[i];
            }
            return kb31_septic_curve_t(x, y);
        }

        static FUN kb31_septic_curve_t start_point() {
            kb31_septic_extension_t x;
            kb31_septic_extension_t y;
            for (uintptr_t i = 0 ; i < 7 ; i++) {
                x.value[i] = start_x[i];
                y.value[i] = start_y[i];
            }
            return kb31_septic_curve_t(x, y);
        }

        FUN bool is_infinity() const {
            return x == kb31_septic_extension_t::zero() && y == kb31_septic_extension_t::zero();
        }

        FUN kb31_septic_curve_t& operator+=(const kb31_septic_curve_t b) {
            if (b.is_infinity()) {
                return *this;
            }
            if (is_infinity()) {
                x = b.x;
                y = b.y;
                return *this;
            }

            kb31_septic_extension_t x_diff = b.x - x;
            if (x_diff == kb31_septic_extension_t::zero()) {
                if (y == b.y) {
                    kb31_septic_extension_t y2 = y + y; 
                    kb31_septic_extension_t x2 = x * x;
                    // FIX (septic-doubling): slope numerator must be 3x^2 + 3z
                    // (curve linear coeff a=3z at coeff index 1), matching host
                    // SepticCurve::double() (septic_curve.rs:69-84) + this header's
                    // own curve_formula (t[1]=kb31_t(int(3))).  The old
                    // `+ kb31_t::two()` wrongly added scalar 2 at coeff index 0.
                    // Latent-only (doubling branch unreachable for honest traces),
                    // device-only -> NO vk regen.
                    kb31_septic_extension_t num = x2 + x2 + x2;
                    num.value[1] = num.value[1] + kb31_t(int(3));
                    kb31_septic_extension_t slope = num / y2;
                    kb31_septic_extension_t result_x = slope * slope - x - x;
                    kb31_septic_extension_t result_y = slope * (x - result_x) - y;
                    x = result_x;
                    y = result_y;
                    return *this;
                }
                else {
                    x = kb31_septic_extension_t::zero();
                    y = kb31_septic_extension_t::zero();
                    return *this;
                }
            }
            else {
                kb31_septic_extension_t slope = (b.y - y) / x_diff;
                kb31_septic_extension_t new_x = slope * slope - x - b.x;
                y = slope * (x - new_x) - y;
                x = new_x;
                return *this;
            }
        }

        friend FUN kb31_septic_curve_t operator+(kb31_septic_curve_t a, const kb31_septic_curve_t b) {
            return a += b;
        }

        static FUN kb31_septic_extension_t sum_checker_x(
            const kb31_septic_curve_t& p1,
            const kb31_septic_curve_t& p2,
            const kb31_septic_curve_t& p3
        ) {
            kb31_septic_extension_t x_diff = p2.x - p1.x;
            kb31_septic_extension_t y_diff = p2.y - p1.y;
            return (p1.x + p2.x + p3.x) * x_diff * x_diff - y_diff * y_diff;
        }
};

class kb31_septic_digest_t {
    public:
        kb31_septic_curve_t point;

        FUN kb31_septic_digest_t() {
            this->point = kb31_septic_curve_t();
        }

        FUN kb31_septic_digest_t(kb31_t value[14]) {
            this->point = kb31_septic_curve_t(value);
        }

        FUN kb31_septic_digest_t(kb31_septic_extension_t x, kb31_septic_extension_t y) {
            this->point = kb31_septic_curve_t(x, y);
        }

        FUN kb31_septic_digest_t(kb31_septic_curve_t point) {
            this->point = point;
        }
};

