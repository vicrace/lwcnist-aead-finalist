#ifndef OPERATION_H_
#define OPERATION_H_

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "params.h"

//Initialise buffer
static void init_buffer(char n, uint8_t* buffer, uint64_t numbytes, int BATCH) {

	for (int k = 0; k < BATCH; k++) {
		//srand(k);
		for (int i = 0; i < numbytes; i++) {
			//srand(k + (uint8_t)(rand() % (numbytes - 0 + 1) + 0));
			//(uint8_t)(rand() % (BATCH - 0 + 1) + 0)
			buffer[k * numbytes + i] = (uint8_t)(rand() % (k - 0 + 1) + 0) + i;

		}
	}
}

void print(uint8_t c, uint8_t* x, uint64_t xlen) {
	uint64_t i;
	printf("%c[%d]=", c, (int)xlen);
	for (i = 0; i < xlen; ++i) printf("%x", x[i]);
	printf("\n");
}


//Check cuda function
static void CHECK(cudaError_t call)
{
	const cudaError_t error = call;
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
		fprintf(stderr, "code: %d, reason: %s\n", error,
			cudaGetErrorString(error));
		exit(1);
	}
}

//check gpu and cpu results
static void checkResult(char* version, uint8_t* host, uint8_t* gpu, uint64_t clen, int BATCH) {

	bool check = true;
	uint64_t i;
	for (int k = 0; k < BATCH-1; k++) {
		for (i = 0; i < clen; ++i) {
			if (host[k * clen + i] != gpu[k * clen + i]) {
				printf("\nVersion %s - \tBatch %d ,element %d not match, host - %x\t gpu - %x\n", version, k, i, host[k * clen + i], gpu[k * clen + i]);
				check = false;
				break;
			}
		}
		if (!check) break;
	}
	if (!check) printf("-> Version %s - \Not Match!!", version);

}


static void PrintTime(uint8_t* ct, uint8_t* h_c, uint64_t* d_clen, int threads, float memcpyH2D, float kernelTime, float memcpyD2H, float cpu_t, FILE * fpt, int BATCH, uint8_t* tag, uint8_t* h_t, char* name ) {

	constexpr size_t tlen = 16ul;

	checkResult("Cipher", ct, h_c, MAX_CIPHER_LENGTH, BATCH);
#ifdef PRINTC
	for (int i = 0; i < BATCH; i++) {
		printf("GPU -> ");
		print('c', h_c + i * (*d_clen), *d_clen);
	}
#endif // PRINT

	size_t size = BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t);
	float total = memcpyH2D + kernelTime + memcpyD2H;
	printf("KernelT%d :\t %.6f ms\t %.2f times\t%.f\t%.2f more \t%s\n", threads, total, (cpu_t / total), BATCH / (total / 1000), (BATCH / (total / 1000)) / (BATCH / (cpu_t / 1000)), name);
#ifdef WRITEFILE
	fprintf(fpt, "T%d,%d,%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.f, %.2f, %s\n", threads, BATCH, (size * 2e-6) / total, total, (cpu_t / total), ((size * 2e-6) / total) * 8, memcpyH2D, kernelTime, (cpu_t / kernelTime), BATCH / (total / 1000), (BATCH / (total / 1000)) / (BATCH / (cpu_t / 1000)), name);
#endif
}
#endif // OPERATION_H_
