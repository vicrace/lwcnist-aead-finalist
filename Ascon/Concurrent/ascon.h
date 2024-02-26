#include "params.h"


#define ROTR_GO(x,n) (((x)>>(n))|((x)<<(64-(n))))

#define ROUND_GO(C) \
    x2 ^= C;\
    x0 ^= x4;\
    x4 ^= x3;\
    x2 ^= x1;\
    t0 = x0;\
    t4 = x4;\
    t3 = x3;\
    t1 = x1;\
    t2 = x2;\
    x0 = t0 ^ ((~t1) & t2);\
    x2 = t2 ^ ((~t3) & t4);\
    x4 = t4 ^ ((~t0) & t1);\
    x1 = t1 ^ ((~t2) & t3);\
    x3 = t3 ^ ((~t4) & t0);\
    x1 ^= x0;\
    t1  = x1;\
    x1 = ROTR_GO(x1, 39);\
    x3 ^= x2;\
    t2  = x2;\
    x2 = ROTR_GO(x2, 1);\
    t4  = x4;\
    t2 ^= x2;\
    x2 = ROTR_GO(x2, 5);\
    t3  = x3;\
    t1 ^= x1;\
    x3 = ROTR_GO(x3, 10);\
    x0 ^= x4;\
    x4 = ROTR_GO(x4, 7);\
    t3 ^= x3;\
    x2 ^= t2;\
    x1 = ROTR_GO(x1, 22);\
    t0  = x0;\
    x2 = ~x2;\
    x3 = ROTR_GO(x3, 7);\
    t4 ^= x4;\
    x4 = ROTR_GO(x4, 34);\
    x3 ^= t3;\
    x1 ^= t1;\
    x0 = ROTR_GO(x0, 19);\
    x4 ^= t4;\
    t0 ^= x0;\
    x0 = ROTR_GO(x0, 9);\
    x0 ^= t0;

#define P12_GO \
    ROUND_GO(0xf0);\
    ROUND_GO(0xe1);\
    ROUND_GO(0xd2);\
    ROUND_GO(0xc3);\
    ROUND_GO(0xb4);\
    ROUND_GO(0xa5);\
    ROUND_GO(0x96);\
    ROUND_GO(0x87);\
    ROUND_GO(0x78);\
    ROUND_GO(0x69);\
    ROUND_GO(0x5a);\
    ROUND_GO(0x4b);


#define P8_GO   \
    ROUND_GO(0xb4); \
    ROUND_GO(0xa5); \
    ROUND_GO(0x96); \
    ROUND_GO(0x87); \
    ROUND_GO(0x78); \
    ROUND_GO(0x69); \
    ROUND_GO(0x5a); \
    ROUND_GO(0x4b);


#define P6_GO \
    ROUND_GO(0x96);\
    ROUND_GO(0x87);\
    ROUND_GO(0x78);\
    ROUND_GO(0x69);\
    ROUND_GO(0x5a);\
    ROUND_GO(0x4b);

#define ROUND_GO_ARR(C) \
    x[2] ^= C;\
    x[0] ^= x[4];\
    x[4] ^= x[3];\
    x[2] ^= x[1];\
    t[0] = x[0];\
    t[4] = x[4];\
    t[3] = x[3];\
    t[1] = x[1];\
    t[2] = x[2];\
    x[0] = t[0] ^ ((~t[1]) & t[2]);\
    x[2] = t[2] ^ ((~t[3]) & t[4]);\
    x[4] = t[4] ^ ((~t[0]) & t[1]);\
    x[1] = t[1] ^ ((~t[2]) & t[3]);\
    x[3] = t[3] ^ ((~t[4]) & t[0]);\
    x[1] ^= x[0];\
    t[1]  = x[1];\
    x[1] = ROTR_GO(x[1], 39);\
    x[3] ^= x[2];\
    t[2]  = x[2];\
    x[2] = ROTR_GO(x[2], 1);\
    t[4]  = x[4];\
    t[2] ^= x[2];\
    x[2] = ROTR_GO(x[2], 5);\
    t[3]  = x[3];\
    t[1] ^= x[1];\
    x[3] = ROTR_GO(x[3], 10);\
    x[0] ^= x[4];\
    x[4] = ROTR_GO(x[4], 7);\
    t[3] ^= x[3];\
    x[2] ^= t[2];\
    x[1] = ROTR_GO(x[1], 22);\
    t[0]  = x[0];\
    x[2] = ~x[2];\
    x[3] = ROTR_GO(x[3], 7);\
    t[4] ^= x[4];\
    x[4] = ROTR_GO(x[4], 34);\
    x[3] ^= t[3];\
    x[1] ^= t[1];\
    x[0] = ROTR_GO(x[0], 19);\
    x[4] ^= t[4];\
    t[0] ^= x[0];\
    x[0] = ROTR_GO(x[0], 9);\
    x[0] ^= t[0];

#define P12_GO_ARR \
    ROUND_GO_ARR(0xf0);\
    ROUND_GO_ARR(0xe1);\
    ROUND_GO_ARR(0xd2);\
    ROUND_GO_ARR(0xc3);\
    ROUND_GO_ARR(0xb4);\
    ROUND_GO_ARR(0xa5);\
    ROUND_GO_ARR(0x96);\
    ROUND_GO_ARR(0x87);\
    ROUND_GO_ARR(0x78);\
    ROUND_GO_ARR(0x69);\
    ROUND_GO_ARR(0x5a);\
    ROUND_GO_ARR(0x4b);

#define P6_GO_ARR \
    ROUND_GO_ARR(0x96);\
    ROUND_GO_ARR(0x87);\
    ROUND_GO_ARR(0x78);\
    ROUND_GO_ARR(0x69);\
    ROUND_GO_ARR(0x5a);\
    ROUND_GO_ARR(0x4b);