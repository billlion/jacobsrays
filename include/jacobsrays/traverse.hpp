// SPDX-License-Identifier: Apache-2.0
#ifndef JACOBSRAYS_TRAVERSE_HPP
#define JACOBSRAYS_TRAVERSE_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace jacobsrays {
using index_t = std::int64_t;

struct Grid {
    index_t n[3];                 // x, y, z; x varies fastest
    double spacing[3], origin[3]; // origin is the LOWER CORNER
    index_t size() const { return n[0] * n[1] * n[2]; }
    void validate() const {
        index_t product = 1;
        for (int a = 0; a < 3; ++a) {
            if (n[a] < 1 || n[a] > (index_t(1) << 52) ||
                product > std::numeric_limits<index_t>::max() / n[a])
                throw std::invalid_argument("invalid or overflowing grid shape");
            product *= n[a];
            const double hi = origin[a] + n[a] * spacing[a];
            if (!std::isfinite(origin[a]) || !std::isfinite(spacing[a]) ||
                spacing[a] <= 0 || !std::isfinite(hi) || hi <= origin[a] ||
                origin[a] + spacing[a] == origin[a] || hi - spacing[a] == hi)
                throw std::invalid_argument("grid spacing/extent is not representable");
        }
    }
};

inline void validate_ray(const Grid& g, const double* p, const double* e) {
    double d[3];
    for (int a = 0; a < 3; ++a) {
        d[a] = e[a] - p[a];
        const double q = (p[a] - g.origin[a]) / g.spacing[a];
        const double v = d[a] / g.spacing[a];
        if (!std::isfinite(p[a]) || !std::isfinite(e[a]) ||
            !std::isfinite(q) || !std::isfinite(v) ||
            (d[a] != 0 && (v == 0 || !std::isfinite(1.0/v))))
            throw std::invalid_argument("ray coordinates exceed supported floating-point range");
    }
    if (!std::isfinite(std::hypot(std::hypot(d[0], d[1]), d[2])))
        throw std::invalid_argument("ray length overflows double precision");
}

// Visit only positive-length voxel intervals, in p -> e order. No allocation.
// Callback: emit(x-fastest zero-based index, physical length, t_enter, t_exit).
// Inputs must be validated before calling. t parameterises p+t*(e-p), 0<=t<=1.
template<class Emit>
inline void walk(const Grid& g, const double* p, const double* e, Emit emit) {
    double q[3], v[3], inv[3], next[3];
    index_t cell[3];
    int step[3];
    double first = 0.0, last = 1.0;
    const double length = std::hypot(std::hypot(e[0]-p[0], e[1]-p[1]), e[2]-p[2]);
    if (length == 0) return;
    for (int a = 0; a < 3; ++a) {
        q[a] = (p[a] - g.origin[a]) / g.spacing[a];
        v[a] = (e[a] - p[a]) / g.spacing[a];
        if (v[a] == 0) {
            // Half-open boundary ownership: internal plane -> higher cell;
            // lower outer face included, upper outer face excluded.
            if (q[a] < 0 || q[a] >= g.n[a]) return;
            inv[a] = 0;
        } else {
            inv[a] = 1.0/v[a];
            double lo = -q[a]*inv[a], hi = (g.n[a]-q[a])*inv[a];
            if (lo > hi) std::swap(lo, hi);
            first = std::max(first, lo);
            last = std::min(last, hi);
        }
    }
    if (first >= last) return;
    for (int a = 0; a < 3; ++a) {
        const double x = std::max(0.0, std::min(double(g.n[a]), std::fma(first, v[a], q[a])));
        cell[a] = static_cast<index_t>(v[a] < 0 ? std::ceil(x)-1 : std::floor(x));
        cell[a] = std::max(index_t(0), std::min(g.n[a]-1, cell[a]));
        step[a] = (v[a] > 0) - (v[a] < 0);
        next[a] = step[a] ? (cell[a] + (step[a] > 0) - q[a])*inv[a]
                             : std::numeric_limits<double>::infinity();
    }
    double t = first;
    while (t < last) {
        const double stop = std::min(last, std::min(next[0], std::min(next[1], next[2])));
        if (stop > t) {
            emit(cell[0] + g.n[0]*(cell[1] + g.n[1]*cell[2]), (stop-t)*length, t, stop);
            t = stop;
        }
        if (stop >= last) break;
        // Advance every tied plane. Recompute from integer plane numbers to
        // avoid accumulated t += dt drift. Near ties retain tiny intervals.
        for (int a = 0; a < 3; ++a) if (next[a] <= stop) {
            cell[a] += step[a];
            if (cell[a] < 0 || cell[a] >= g.n[a]) return;
            next[a] = (cell[a] + (step[a] > 0) - q[a])*inv[a];
        }
    }
}
} // namespace jacobsrays
#endif
