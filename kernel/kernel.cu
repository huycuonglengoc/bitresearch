// secp256k1 private-key search kernel (CUDA).
//
// Field arithmetic: 256-bit numbers as 8 x uint32 little-endian limbs. The
// 512-bit product of a multiplication is reduced mod p using the identity
// 2^256 == 2^32 + 977 (mod p) for p = 2^256 - 2^32 - 977.
//
// Point arithmetic: Jacobian coordinates with mixed (Jacobian + affine)
// addition. The base point / step point / table are passed in as affine points
// (precomputed on the host), so the kernel never needs a scalar multiplication
// beyond summing the `2^i * G` table.
//
// Per-key affine conversion (needed to hash) uses a shared-memory Montgomery
// batch inversion: one field inverse per block instead of one per thread.

typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;

// ---------------------------------------------------------------------------
// Field element (8 x u32, little-endian)
// ---------------------------------------------------------------------------
typedef struct {
    u32 n[8];
} fe;

// p = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
__device__ __constant__ fe FE_P = {{
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
}};

// Generator G (affine).
__device__ __constant__ fe GX = {{
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
}};
__device__ __constant__ fe GY = {{
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
}};

__device__ __forceinline__ void fe_zero(fe *r) {
    for (int i = 0; i < 8; i++) r->n[i] = 0;
}

__device__ __forceinline__ int fe_is_zero(const fe *a) {
    u32 t = 0;
    for (int i = 0; i < 8; i++) t |= a->n[i];
    return t == 0;
}

__device__ __forceinline__ void fe_from_u32(fe *r, u32 v) {
    r->n[0] = v;
    for (int i = 1; i < 8; i++) r->n[i] = 0;
}

__device__ __forceinline__ void fe_copy(fe *r, const fe *a) {
    for (int i = 0; i < 8; i++) r->n[i] = a->n[i];
}

__device__ __forceinline__ void fe_load(fe *r, const u32 *p) {
    for (int i = 0; i < 8; i++) r->n[i] = p[i];
}

__device__ __forceinline__ void fe_store(u32 *p, const fe *r) {
    for (int i = 0; i < 8; i++) p[i] = r->n[i];
}

// a >= b ?
__device__ __forceinline__ int fe_ge(const fe *a, const fe *b) {
    for (int i = 7; i >= 0; i--) {
        if (a->n[i] > b->n[i]) return 1;
        if (a->n[i] < b->n[i]) return 0;
    }
    return 1;
}

// r = a - b (raw, assumes a >= b). No reduction.
__device__ __forceinline__ void fe_sub_raw(fe *r, const fe *a, const fe *b) {
    u64 borrow = 0;
    for (int i = 0; i < 8; i++) {
        u64 d = (u64)a->n[i] - (u64)b->n[i] - borrow;
        r->n[i] = (u32)d;
        borrow = (d >> 32) & 1;
    }
}

// r += c, where c = 2^32 + 977 (the "fold" constant, since p = 2^256 - c).
__device__ __forceinline__ void fe_add_c(fe *r) {
    u64 s0 = (u64)r->n[0] + 977u;
    r->n[0] = (u32)s0;
    u64 carry = s0 >> 32;
    for (int i = 1; i < 8; i++) {
        u64 add = (i == 1) ? (1u + carry) : carry;
        u64 s = (u64)r->n[i] + add;
        r->n[i] = (u32)s;
        carry = s >> 32;
    }
}

// r -= c, where c = 2^32 + 977. Caller guarantees r >= c.
__device__ __forceinline__ void fe_sub_c(fe *r) {
    u64 d0 = (u64)r->n[0] - 977u;
    r->n[0] = (u32)d0;
    u64 borrow = (d0 >> 32) & 1;
    for (int i = 1; i < 8; i++) {
        u64 sub = (i == 1) ? (1u + borrow) : borrow;
        u64 d = (u64)r->n[i] - sub;
        r->n[i] = (u32)d;
        borrow = (d >> 32) & 1;
    }
}

// r = a + b mod p. a + b < 2p.
__device__ __forceinline__ void fe_add(fe *r, const fe *a, const fe *b) {
    u64 carry = 0;
    for (int i = 0; i < 8; i++) {
        u64 s = (u64)a->n[i] + (u64)b->n[i] + carry;
        r->n[i] = (u32)s;
        carry = s >> 32;
    }
    if (carry) {
        // sum = r + 2^256 >= p; sum - p = r + c, and this is < p.
        fe_add_c(r);
    } else if (fe_ge(r, &FE_P)) {
        fe_sub_raw(r, r, &FE_P);
    }
}

// r = a - b mod p.
__device__ __forceinline__ void fe_sub(fe *r, const fe *a, const fe *b) {
    u64 borrow = 0;
    for (int i = 0; i < 8; i++) {
        u64 d = (u64)a->n[i] - (u64)b->n[i] - borrow;
        r->n[i] = (u32)d;
        borrow = (d >> 32) & 1;
    }
    if (borrow) {
        // a < b: r = a - b + 2^256; a - b + p = r - c (and this is in (0, p)).
        fe_sub_c(r);
    }
}

// Fold a 16-limb (512-bit) value into 8 limbs mod p.
__device__ __noinline__ void fe_reduce(fe *r, const u32 *t16) {
    u32 tmp[16];
    for (int i = 0; i < 16; i++) tmp[i] = t16[i];

    for (int pass = 0; pass < 8; pass++) {
        int dirty = 0;
        for (int k = 15; k >= 8; k--) {
            if (tmp[k] != 0) {
                u32 v = tmp[k];
                tmp[k] = 0;
                // add v * 977 at position k-8
                {
                    u64 acc = (u64)v * 977u;
                    for (int i = k - 8; acc != 0 && i < 16; i++) {
                        u64 s = (u64)tmp[i] + (u32)acc;
                        tmp[i] = (u32)s;
                        acc = (acc >> 32) + (s >> 32);
                    }
                }
                // add v at position k-7 (the *2^32 part)
                {
                    u64 acc = (u64)v;
                    for (int i = k - 7; acc != 0 && i < 16; i++) {
                        u64 s = (u64)tmp[i] + (u32)acc;
                        tmp[i] = (u32)s;
                        acc = (acc >> 32) + (s >> 32);
                    }
                }
                dirty = 1;
            }
        }
        if (!dirty) break;
    }

    for (int i = 0; i < 8; i++) r->n[i] = tmp[i];
    while (fe_ge(r, &FE_P)) {
        fe_sub_raw(r, r, &FE_P);
    }
}

// r = a * b mod p
//
// NOTE: `__noinline__` (not `__forceinline__`). Forcing every field op to inline
// into the search kernel makes nvcc/cicc build one giant function whose codegen
// hangs the compiler (it did, at every -O level). Keeping the heavy arithmetic
// as separate device functions compiles instantly; the call overhead is small.
__device__ __noinline__ void fe_mul(fe *r, const fe *a, const fe *b) {
    u32 t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 8; i++) {
        u64 carry = 0;
        for (int j = 0; j < 8; j++) {
            u64 acc = (u64)a->n[i] * (u64)b->n[j] + (u64)t[i + j] + carry;
            t[i + j] = (u32)acc;
            carry = acc >> 32;
        }
        t[i + 8] = (u32)carry;
    }
    fe_reduce(r, t);
}

// r = a^2 mod p
__device__ __noinline__ void fe_sqr(fe *r, const fe *a) {
    fe_mul(r, a, a);
}

// r = a^(p-2) mod p (Fermat inverse). exp = p - 2.
//
// IMPORTANT: `__noinline__` + `#pragma unroll 1` are deliberate. The loop has a
// constant bound and a compile-time-constant exponent, so without these hints
// nvcc/ptxas fully unrolls 256 iterations of fe_sqr/fe_mul into the
// (forceinlined) caller — a code explosion that makes the compiler hang for
// minutes. Compiled once as a normal function, compile time is instant; the
// runtime cost (256 sqr + ~128 mul per inverse) is amortized by batch inversion
// (a single inverse per block).
__device__ __noinline__ void fe_inv(fe *r, const fe *a) {
    const u32 exp[8] = {
        0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };
    fe result;
    fe_from_u32(&result, 1);
    #pragma unroll 1
    for (int bit = 255; bit >= 0; bit--) {
        fe_sqr(&result, &result);
        if ((exp[bit >> 5] >> (bit & 31)) & 1) {
            fe_mul(&result, &result, a);
        }
    }
    fe_copy(r, &result);
}

// Big-endian 32-byte encoding (for hashing).
__device__ __forceinline__ void fe_to_be(u32 *out, const fe *a) {
    // out[0] = most significant 32-bit word.
    for (int i = 0; i < 8; i++) {
        out[7 - i] = a->n[i];
    }
}

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------
typedef struct {
    fe x, y;
} affine;

typedef struct {
    fe x, y, z; // z == 0 encodes the point at infinity
} jacobian;

// P + Q -> r, where P is Jacobian, Q affine. Handles P at infinity.
__device__ __noinline__ void point_add_mixed(jacobian *r, const jacobian *p, const affine *q) {
    if (fe_is_zero(&p->z)) {
        fe_copy(&r->x, &q->x);
        fe_copy(&r->y, &q->y);
        fe_from_u32(&r->z, 1);
        return;
    }
    fe z1z1, u2, s2, h, hh, i, j, rr, v, t1, t2, t3;
    fe_sqr(&z1z1, &p->z);
    fe_mul(&u2, &q->x, &z1z1);
    fe_mul(&s2, &q->y, &p->z);
    fe_mul(&s2, &s2, &z1z1);
    fe_sub(&h, &u2, &p->x);
    fe_sqr(&hh, &h);
    fe_add(&i, &hh, &hh);   // 2*hh
    fe_add(&i, &i, &i);     // 4*hh
    fe_mul(&j, &h, &i);
    fe_sub(&rr, &s2, &p->y);
    fe_add(&rr, &rr, &rr);  // 2*(s2 - y1)
    fe_mul(&v, &p->x, &i);
    // x3 = rr^2 - j - 2v
    fe_sqr(&r->x, &rr);
    fe_sub(&r->x, &r->x, &j);
    fe_add(&t1, &v, &v);
    fe_sub(&r->x, &r->x, &t1);
    // y3 = rr*(v - x3) - 2*y1*j
    fe_sub(&t2, &v, &r->x);
    fe_mul(&r->y, &rr, &t2);
    fe_mul(&t3, &p->y, &j);
    fe_add(&t3, &t3, &t3);
    fe_sub(&r->y, &r->y, &t3);
    // z3 = (z1 + h)^2 - z1z1 - hh
    fe_add(&r->z, &p->z, &h);
    fe_sqr(&r->z, &r->z);
    fe_sub(&r->z, &r->z, &z1z1);
    fe_sub(&r->z, &r->z, &hh);
}

// key * G in Jacobian coordinates, using the 2^i * G table.
__device__ __noinline__ void point_from_u64(jacobian *out, const u32 *g_tab, u64 key) {
    fe_zero(&out->z);
    #pragma unroll 1  // 64 iterations of point_add_mixed; unrolling inflates codegen
    for (int i = 0; i < 64; i++) {
        if ((key >> i) & 1ULL) {
            affine q;
            fe_load(&q.x, &g_tab[(i * 2 + 0) * 8]);
            fe_load(&q.y, &g_tab[(i * 2 + 1) * 8]);
            point_add_mixed(out, out, &q);
        }
    }
}

// ---------------------------------------------------------------------------
// SHA-256 (single 64-byte block)
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 rotr32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ __constant__ u32 SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __noinline__ void sha256_compress(u32 *out, const u8 *msg) {
    u32 w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((u32)msg[4 * i] << 24) | ((u32)msg[4 * i + 1] << 16) |
               ((u32)msg[4 * i + 2] << 8) | (u32)msg[4 * i + 3];
    }
    for (int i = 16; i < 64; i++) {
        u32 s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        u32 s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    u32 a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a,
        e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
    for (int i = 0; i < 64; i++) {
        u32 S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        u32 ch = (e & f) ^ (~e & g);
        u32 t1 = h + S1 + ch + SHA256_K[i] + w[i];
        u32 S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        u32 maj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    u32 H[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    for (int i = 0; i < 8; i++) out[i] = H[i] + (i == 0 ? a : i == 1 ? b : i == 2 ? c : i == 3 ? d : i == 4 ? e : i == 5 ? f : i == 6 ? g : h);
}

// ---------------------------------------------------------------------------
// RIPEMD-160 (single 64-byte block)
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 rotl32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ __forceinline__ u32 ripemd_f(int r, u32 x, u32 y, u32 z) {
    if (r == 0) return x ^ y ^ z;
    if (r == 1) return (x & y) | (~x & z);
    if (r == 2) return (x | ~y) ^ z;
    if (r == 3) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

__device__ __constant__ u32 RIPEMD_KL[5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
__device__ __constant__ u32 RIPEMD_KR[5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};

__device__ __constant__ int RIPEMD_RL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

__device__ __constant__ int RIPEMD_RR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

__device__ __constant__ int RIPEMD_SL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

__device__ __constant__ int RIPEMD_SR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

__device__ __noinline__ void ripemd160_compress(u32 *out, const u8 *msg) {
    u32 x[16];
    for (int i = 0; i < 16; i++) {
        x[i] = ((u32)msg[4 * i] << 24) | ((u32)msg[4 * i + 1] << 16) |
               ((u32)msg[4 * i + 2] << 8) | (u32)msg[4 * i + 3];
    }

    u32 al = 0x67452301, bl = 0xefcdab89, cl = 0x98badcfe, dl = 0x10325476, el = 0xc3d2e1f0;
    u32 ar = 0x67452301, br = 0xefcdab89, cr = 0x98badcfe, dr = 0x10325476, er = 0xc3d2e1f0;

    for (int j = 0; j < 80; j++) {
        int rnd = j >> 4;
        u32 t = rotl32(al + ripemd_f(rnd, bl, cl, dl) + x[RIPEMD_RL[j]] + RIPEMD_KL[rnd], RIPEMD_SL[j]) + el;
        al = el; el = dl; dl = rotl32(cl, 10); cl = bl; bl = t;
    }
    for (int j = 0; j < 80; j++) {
        int rnd = 4 - (j >> 4);
        u32 t = rotl32(ar + ripemd_f(rnd, br, cr, dr) + x[RIPEMD_RR[j]] + RIPEMD_KR[rnd], RIPEMD_SR[j]) + er;
        ar = er; er = dr; dr = rotl32(cr, 10); cr = br; br = t;
    }

    u32 t = 0xefcdab89 + cl + dr; out[1] = t;
    t = 0x98badcfe + dl + er; out[2] = t;
    t = 0x10325476 + el + ar; out[3] = t;
    t = 0xc3d2e1f0 + al + br; out[4] = t;
    t = 0x67452301 + bl + cr; out[0] = t;
}

// ---------------------------------------------------------------------------
// HASH160 of a compressed public key
// ---------------------------------------------------------------------------
__device__ __noinline__ void hash160_from_point(const jacobian *P, u32 *out5) {
    fe zinv, zinv2, zinv3, ax, ay;
    fe_inv(&zinv, &P->z);
    fe_sqr(&zinv2, &zinv);
    fe_mul(&ax, &P->x, &zinv2);
    fe_mul(&zinv3, &zinv2, &zinv);
    fe_mul(&ay, &P->y, &zinv3);
    u32 parity = ay.n[0] & 1;

    u8 msg[64];
    msg[0] = parity ? 0x03 : 0x02;
    u32 xbe[8];
    fe_to_be(xbe, &ax);
    for (int i = 0; i < 8; i++) {
        msg[1 + 4 * i + 0] = (xbe[i] >> 24) & 0xff;
        msg[1 + 4 * i + 1] = (xbe[i] >> 16) & 0xff;
        msg[1 + 4 * i + 2] = (xbe[i] >> 8) & 0xff;
        msg[1 + 4 * i + 3] = xbe[i] & 0xff;
    }
    msg[33] = 0x80;
    for (int i = 34; i < 56; i++) msg[i] = 0;
    msg[56] = 0; msg[57] = 0; msg[58] = 0; msg[59] = 0;
    msg[60] = 0; msg[61] = 0; msg[62] = 0x01; msg[63] = 0x08; // 264 bits

    u32 h[8];
    sha256_compress(h, msg);

    u8 rmsg[64];
    for (int i = 0; i < 8; i++) {
        rmsg[4 * i + 0] = (h[i] >> 24) & 0xff;
        rmsg[4 * i + 1] = (h[i] >> 16) & 0xff;
        rmsg[4 * i + 2] = (h[i] >> 8) & 0xff;
        rmsg[4 * i + 3] = h[i] & 0xff;
    }
    rmsg[32] = 0x80;
    for (int i = 33; i < 56; i++) rmsg[i] = 0;
    rmsg[56] = 0; rmsg[57] = 0; rmsg[58] = 0; rmsg[59] = 0;
    rmsg[60] = 0; rmsg[61] = 0; rmsg[62] = 0x01; rmsg[63] = 0x00; // 256 bits

    ripemd160_compress(out5, rmsg);
}

// ---------------------------------------------------------------------------
// Batch inversion (shared memory). zorig/prod are n*8 u32 each.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void batch_invert(u32 *zorig, u32 *prod, const fe *z, int tid, int n, fe *zinv) {
    for (int i = 0; i < 8; i++) {
        zorig[tid * 8 + i] = z->n[i];
        prod[tid * 8 + i] = z->n[i];
    }
    __syncthreads();
    if (tid == 0) {
        fe acc, cur, prev, zt, r;
        fe_load(&acc, &prod[0]);
        for (int t = 1; t < n; t++) {
            fe_load(&cur, &prod[t * 8]);
            fe_mul(&acc, &acc, &cur);
            fe_store(&prod[t * 8], &acc);
        }
        fe_inv(&acc, &acc);
        for (int t = n - 1; t >= 1; t--) {
            fe_load(&prev, &prod[(t - 1) * 8]);
            fe_mul(&r, &acc, &prev);
            fe_store(&prod[t * 8], &r);
            fe_load(&zt, &zorig[t * 8]);
            fe_mul(&acc, &acc, &zt);
        }
        fe_store(&prod[0], &acc);
    }
    __syncthreads();
    fe_load(zinv, &prod[tid * 8]);
}

// ---------------------------------------------------------------------------
// Search kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void search_kernel(
    const u32 *g_tab,        // 64 * 2 * 8 u32: affine 2^i * G
    const u32 *p_start,      // 2 * 8 u32: affine start*G
    const u32 *step_pt,      // 2 * 8 u32: affine (T * G)
    const u32 *target,       // 5 u32: HASH160
    u64 start,
    u64 end,
    u64 *found_key,
    u32 *found_flag)
{
    int tid = threadIdx.x;
    int n = blockDim.x;
    u64 T = (u64)gridDim.x * (u64)blockDim.x;
    u64 idx = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;

    // P = idx * G
    jacobian P;
    point_from_u64(&P, g_tab, idx);

    // P += p_start
    affine q;
    fe_load(&q.x, &p_start[0]);
    fe_load(&q.y, &p_start[8]);
    point_add_mixed(&P, &P, &q);

    affine step;
    fe_load(&step.x, &step_pt[0]);
    fe_load(&step.y, &step_pt[8]);

    extern __shared__ u32 shm[];
    u32 *zorig = shm;
    u32 *prod = shm + n * 8;

    // Uniform iteration count for the whole block: thread 0 of this block has
    // the smallest key, hence the most iterations. Every thread runs the same
    // number of iterations so the __syncthreads() inside batch_invert never
    // diverges (which would deadlock on partial blocks).
    u64 block_base = start + (idx - tid); // key of thread 0 in this block
    u64 n_iter = 0;
    if (block_base <= end) {
        n_iter = (end - block_base) / T + 1;
    }

    for (u64 it = 0; it < n_iter; it++) {
        u64 key = start + idx + it * T;
        int in_range = (key <= end);

        fe zinv;
        batch_invert(zorig, prod, &P.z, tid, n, &zinv);

        fe zinv2, zinv3, ax, ay;
        fe_sqr(&zinv2, &zinv);
        fe_mul(&ax, &P.x, &zinv2);
        fe_mul(&zinv3, &zinv2, &zinv);
        fe_mul(&ay, &P.y, &zinv3);
        u32 parity = ay.n[0] & 1;

        u8 msg[64];
        msg[0] = parity ? 0x03 : 0x02;
        u32 xbe[8];
        fe_to_be(xbe, &ax);
        for (int i = 0; i < 8; i++) {
            msg[1 + 4 * i + 0] = (xbe[i] >> 24) & 0xff;
            msg[1 + 4 * i + 1] = (xbe[i] >> 16) & 0xff;
            msg[1 + 4 * i + 2] = (xbe[i] >> 8) & 0xff;
            msg[1 + 4 * i + 3] = xbe[i] & 0xff;
        }
        msg[33] = 0x80;
        for (int i = 34; i < 56; i++) msg[i] = 0;
        msg[56] = 0; msg[57] = 0; msg[58] = 0; msg[59] = 0;
        msg[60] = 0; msg[61] = 0; msg[62] = 0x01; msg[63] = 0x08;

        u32 h[8];
        sha256_compress(h, msg);
        u8 rmsg[64];
        for (int i = 0; i < 8; i++) {
            rmsg[4 * i + 0] = (h[i] >> 24) & 0xff;
            rmsg[4 * i + 1] = (h[i] >> 16) & 0xff;
            rmsg[4 * i + 2] = (h[i] >> 8) & 0xff;
            rmsg[4 * i + 3] = h[i] & 0xff;
        }
        rmsg[32] = 0x80;
        for (int i = 33; i < 56; i++) rmsg[i] = 0;
        rmsg[56] = 0; rmsg[57] = 0; rmsg[58] = 0; rmsg[59] = 0;
        rmsg[60] = 0; rmsg[61] = 0; rmsg[62] = 0x01; rmsg[63] = 0x00;

        u32 h160[5];
        ripemd160_compress(h160, rmsg);

        if (in_range && h160[0] == target[0] && h160[1] == target[1] &&
            h160[2] == target[2] && h160[3] == target[3] && h160[4] == target[4]) {
            if (atomicCAS(found_flag, 0u, 1u) == 0u) {
                *found_key = key;
            }
        }

        // P += step
        point_add_mixed(&P, &P, &step);
    }
}

// ---------------------------------------------------------------------------
// Self-test kernel: compute HASH160 for each test key.
// ---------------------------------------------------------------------------
extern "C" __global__ void selftest_kernel(
    const u32 *g_tab,
    const u64 *keys,
    u32 *out_hash, // 5 u32 per key
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 key = keys[i];

    jacobian P;
    point_from_u64(&P, g_tab, key);
    hash160_from_point(&P, &out_hash[i * 5]);
}
