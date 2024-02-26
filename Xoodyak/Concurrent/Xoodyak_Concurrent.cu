#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <string.h>
#include <math.h>
#include "xoodyakCPU.h"
#include "xoodyakGPU.h"
#include "params.h"
#include "operations.h"


int crypto_aead_encrypt(
	unsigned char* c, unsigned long long* clen,
	const unsigned char* m, unsigned long long mlen,
	const unsigned char* ad, unsigned long long adlen,
	const unsigned char* nsec,
	const unsigned char* npub,
	const unsigned char* k)
{
	Xoodyak_Instance    instance;

	(void)nsec;

	Xoodyak_Initialize(&instance, k, CRYPTO_KEYBYTES, npub, CRYPTO_NPUBBYTES, NULL, 0);
	Xoodyak_Absorb(&instance, ad, (size_t)adlen);
	Xoodyak_Encrypt(&instance, m, c, (size_t)mlen);
	Xoodyak_Squeeze(&instance, c + mlen, CRYPTO_ABYTES);
	*clen = mlen + CRYPTO_ABYTES;

	return 0;
}
__global__ void crypto_aead_encrypt_gpu_global(
	uint8_t* c, uint64_t* clen,
	const uint8_t* m, uint64_t mlen,
	const uint8_t* ad, uint64_t adlen,
	const uint8_t* nsec, const uint8_t* npub, const uint8_t* k)
{
	Xoodyak_Instance    instance;

	(void)nsec;

	int tid = threadIdx.x, bid = blockIdx.x;

	if ((bid * blockDim.x + tid) < BATCH) {
		uint8_t* C = c + (bid * blockDim.x * (*clen) + tid * (*clen));
		uint8_t* T = c + mlen + (bid * blockDim.x * (*clen) + tid * (*clen));
		const uint8_t* M = m + (bid * blockDim.x * mlen + (tid * mlen));
		const uint8_t* A = ad + (bid * blockDim.x * adlen + tid * adlen);
		const uint8_t* N = npub + (bid * blockDim.x * CRYPTO_KEYBYTES + tid * CRYPTO_KEYBYTES);
		const uint8_t* K = k + (bid * blockDim.x * CRYPTO_KEYBYTES + tid * CRYPTO_KEYBYTES);

		Xoodyak_InitializeG(&instance, K, CRYPTO_KEYBYTES, N, CRYPTO_NPUBBYTES, NULL, 0);
		Xoodyak_AbsorbG(&instance, A, (size_t)adlen);
		Xoodyak_EncryptG(&instance, M, C, (size_t)mlen);
		Xoodyak_SqueezeAnyG(&instance, C + mlen, CRYPTO_ABYTES, 0x40);
		*clen = mlen + CRYPTO_ABYTES;
	}

}

__global__ void crypto_aead_encrypt_gpu_global_Op(
	uint8_t* c, uint64_t* clen,
	const uint8_t* m, uint64_t mlen,
	const uint8_t* ad, uint64_t adlen,
	const uint8_t* nsec, const uint8_t* npub, const uint8_t* k)
{
	Xoodyak_Instance    instance;

	(void)nsec;

	int tid = threadIdx.x, bid = blockIdx.x;
	uint8_t* C = c + (bid * blockDim.x * (*clen) + tid * (*clen));
	uint8_t* T = c + mlen + (bid * blockDim.x * (*clen) + tid * (*clen));
	const uint8_t* M = m + (bid * blockDim.x * mlen + (tid * mlen));
	const uint8_t* A = ad + (bid * blockDim.x * adlen + tid * adlen);
	const uint8_t* N = npub + (bid * blockDim.x * CRYPTO_NPUBBYTES + tid * CRYPTO_NPUBBYTES);
	const uint8_t* K = k + (bid * blockDim.x * CRYPTO_KEYBYTES + tid * CRYPTO_KEYBYTES);

	Xoodyak_InitializeG_Op(&instance, K, CRYPTO_KEYBYTES, N, CRYPTO_NPUBBYTES, NULL, 0);
	Xoodyak_AbsorbG_Op(&instance, A, (size_t)adlen);
	Xoodyak_EncryptG_Op(&instance, M, C, (size_t)mlen);
	Xoodyak_SqueezeAnyG_Op(&instance, C + mlen, CRYPTO_ABYTES, 0x40);
	*clen = mlen + CRYPTO_ABYTES;

}

__global__ void crypto_aead_encrypt_gpu_rcwr_GpuTranspose(
	uint8_t* c, uint64_t* clen,
	const uint8_t* m, uint64_t mlen,
	const uint8_t* ad, uint64_t adlen,
	const uint8_t* nsec, const uint8_t* npub, const uint8_t* k) {

	if ((threadIdx.y * blockDim.x + threadIdx.x) < BATCH) {
		/* Determine matrix index for each data*/
		uint32_t tkix = blockDim.x * blockIdx.x * CRYPTO_KEYBYTES + threadIdx.x;		//For Nonce and key - same because both 16 fixed
		uint32_t tkiy = blockDim.x * blockIdx.x * CRYPTO_KEYBYTES + threadIdx.y;
		uint32_t tmix = blockDim.x * blockIdx.x * mlen + threadIdx.y;					//for message with message len
		uint32_t tmiy = blockDim.x * blockIdx.x * mlen + threadIdx.y;
		uint32_t taix = blockDim.x * blockIdx.x * adlen + threadIdx.x;					//for additional data len
		uint32_t taiy = blockDim.x * blockIdx.x * adlen + threadIdx.y;
		uint32_t tcix = blockDim.x * blockIdx.x * (*clen) + threadIdx.x;				//for cipher text
		uint32_t tciy = blockDim.x * blockIdx.x * (*clen) + threadIdx.y;

		//read in col , write in row
		uint32_t tki = tkiy * CRYPTO_KEYBYTES + tkix; // access in rows - key & nonce
		uint32_t tko = tkix * CRYPTO_KEYBYTES + tkiy; // access in columns - key & nonce
		uint32_t tmi = tmiy * mlen + tmix; // access in rows - message 
		uint32_t tmo = tmix * mlen + tmiy; // access in columns - message 
		uint32_t tai = taiy * adlen + taix; // access in columns - ad 
		uint32_t tao = taix * adlen + taiy; // access in columns - ad 
		uint32_t tci = tciy * (*clen) + tcix; // access in row  - cipher

		//temporarily buffer
		uint8_t* kout = const_cast<uint8_t*>(k) + blockIdx.x * blockDim.x;
		uint8_t* nout = const_cast<uint8_t*>(npub) + blockIdx.x * blockDim.x;
		uint8_t* mout = const_cast<uint8_t*>(m) + blockIdx.x * blockDim.x;
		uint8_t* aout = const_cast<uint8_t*>(ad) + blockIdx.x * blockDim.x;

		kout[tko] = k[tki]; // transpose from row to col for key
		nout[tko] = npub[tki]; //for nonce
		mout[tmo] = m[tmi]; //for message
		aout[tao] = ad[tai]; //for additional data

		__syncthreads();

		uint8_t* C = c + tci;
		const uint8_t* M = m + tmo;
		const uint8_t* A = ad + tao;
		const uint8_t* N = npub + tko;
		const uint8_t* K = k + tko;

		Xoodyak_Instance    instance;

		(void)nsec;

		Xoodyak_InitializeG(&instance, K, CRYPTO_KEYBYTES, N, CRYPTO_NPUBBYTES, NULL, 0);
		Xoodyak_AbsorbG(&instance, A, (size_t)adlen);
		Xoodyak_EncryptG(&instance, M, C, (size_t)mlen);
		Xoodyak_SqueezeAnyG(&instance, C + mlen, CRYPTO_ABYTES, 0x40);
		*clen = mlen + CRYPTO_ABYTES;
	}
}


__inline__ __device__ void encrypt_unroll4(uint8_t* c, uint64_t* clen, const uint8_t* m, uint64_t mlen, const uint8_t* ad, uint64_t adlen,
	const uint8_t* nsec, const uint8_t* npub, const uint8_t* k, uint32_t tko, uint32_t tao, uint32_t tmo, uint32_t tci) {

	Xoodyak_Instance    instance;
	(void)nsec;

	Xoodyak_InitializeG(&instance, k + tko, CRYPTO_KEYBYTES, npub + tko, CRYPTO_NPUBBYTES, NULL, 0);
	Xoodyak_AbsorbG(&instance, ad + tao, (size_t)adlen);
	Xoodyak_EncryptG(&instance, m + tmo, c + tci, (size_t)mlen);
	Xoodyak_SqueezeAnyG(&instance, (c + tci) + mlen, CRYPTO_ABYTES, 0x40);
	*clen = mlen + CRYPTO_ABYTES;
}

__global__ void crypto_aead_encrypt_gpu_rcwr_GPUTransposeUnroll4(
	uint8_t* c, uint64_t* clen,
	const uint8_t* m, uint64_t mlen,
	const uint8_t* ad, uint64_t adlen,
	const uint8_t* nsec, const uint8_t* npub, const uint8_t* k) {

	if ((threadIdx.y * blockDim.x + threadIdx.x) < BATCH) {
		/* Determine matrix index for each data*/
		uint32_t tkix = blockDim.x * blockIdx.x * CRYPTO_KEYBYTES + threadIdx.x;		//For Nonce and key - same because both 16 fixed
		uint32_t tkiy = blockDim.x * blockIdx.x * CRYPTO_KEYBYTES + threadIdx.y;
		uint32_t tmix = blockDim.x * blockIdx.x * mlen + threadIdx.y;					//for message with message len
		uint32_t tmiy = blockDim.x * blockIdx.x * mlen + threadIdx.y;
		uint32_t taix = blockDim.x * blockIdx.x * adlen + threadIdx.x;					//for additional data len
		uint32_t taiy = blockDim.x * blockIdx.x * adlen + threadIdx.y;
		uint32_t tcix = blockDim.x * blockIdx.x * (*clen) + threadIdx.x;				//for cipher text
		uint32_t tciy = blockDim.x * blockIdx.x * (*clen) + threadIdx.y;

		//read in col , write in row
		uint32_t tki = tkiy * CRYPTO_KEYBYTES + tkix; // access in rows - key & nonce
		uint32_t tko = tkix * CRYPTO_KEYBYTES + tkiy; // access in columns - key & nonce
		uint32_t tmi = tmiy * mlen + tmix; // access in rows - message 
		uint32_t tmo = tmix * mlen + tmiy; // access in columns - message 
		uint32_t tai = taiy * adlen + taix; // access in columns - ad 
		uint32_t tao = taix * adlen + taiy; // access in columns - ad 
		uint32_t tci = tciy * (*clen) + tcix; // access in row  - cipher

		//temporarily buffer
		uint8_t* kout = const_cast<uint8_t*>(k) + blockIdx.x * blockDim.x;
		uint8_t* nout = const_cast<uint8_t*>(npub) + blockIdx.x * blockDim.x;
		uint8_t* mout = const_cast<uint8_t*>(m) + blockIdx.x * blockDim.x;
		uint8_t* aout = const_cast<uint8_t*>(ad) + blockIdx.x * blockDim.x;

		kout[tko] = k[tki];													kout[tko + blockDim.x] = k[tki + blockDim.x];
		kout[tko + 2 * blockDim.x] = k[tki + 2 * blockDim.x];				kout[tko + 3 * blockDim.x] = k[tki + 3 * blockDim.x];

		nout[tko] = npub[tki];												nout[tko + blockDim.x] = npub[tki + blockDim.x];
		nout[tko + 2 * blockDim.x] = npub[tki + 2 * blockDim.x];			nout[tko + 3 * blockDim.x] = npub[tki + 3 * blockDim.x];

		mout[tmo] = m[tmi];													mout[tmo + blockDim.x] = m[tmi + blockDim.x];
		mout[tmo + 2 * blockDim.x] = m[tmi + 2 * blockDim.x];				mout[tmo + 3 * blockDim.x] = m[tmi + 3 * blockDim.x];

		aout[tao] = ad[tai];												aout[tao + blockDim.x] = ad[tai + blockDim.x];
		aout[tao + 2 * blockDim.x] = ad[tai + 2 * blockDim.x];				aout[tao + 3 * blockDim.x] = ad[tai + 3 * blockDim.x];

		__syncthreads();

		encrypt_unroll4(c, clen, mout, mlen, aout, adlen, nsec, nout, kout, tko, tao, tmo, tci);
		encrypt_unroll4(c, clen, mout, mlen, aout, adlen, nsec, nout, kout, tko + blockDim.x, tao + blockDim.x, tmo + blockDim.x, tci + blockDim.x);
		encrypt_unroll4(c, clen, mout, mlen, aout, adlen, nsec, nout, kout, tko + 2 * blockDim.x, tao + 2 * blockDim.x, tmo + 2 * blockDim.x, tci + 2 * blockDim.x);
		encrypt_unroll4(c, clen, mout, mlen, aout, adlen, nsec, nout, kout, tko + 3 * blockDim.x, tao + 3 * blockDim.x, tmo + 3 * blockDim.x, tci + 3 * blockDim.x);
	}
}

__global__ void crypto_aead_encrypt_gpu_rcwr_GpuTranspose_Op(
	uint8_t* c, uint64_t* clen,
	const uint8_t* m, uint64_t mlen,
	const uint8_t* ad, uint64_t adlen,
	const uint8_t* nsec, const uint8_t* npub, const uint8_t* k)
{
	/* Determine matrix index for each data*/
	uint32_t tkix = blockDim.x * blockIdx.x * CRYPTO_KEYBYTES + threadIdx.x;		//For Nonce and key - same because both 16 fixed
	uint32_t tkiy = blockDim.x * blockIdx.x * CRYPTO_KEYBYTES + threadIdx.y;
	uint32_t tnix = blockDim.x * blockIdx.x * CRYPTO_NPUBBYTES + threadIdx.x;		//For Nonce and key - same because both 16 fixed
	uint32_t tniy = blockDim.x * blockIdx.x * CRYPTO_NPUBBYTES + threadIdx.y;
	uint32_t tmix = blockDim.x * blockIdx.x * mlen + threadIdx.y;					//for message with message len
	uint32_t tmiy = blockDim.x * blockIdx.x * mlen + threadIdx.y;
	uint32_t taix = blockDim.x * blockIdx.x * adlen + threadIdx.x;					//for additional data len
	uint32_t taiy = blockDim.x * blockIdx.x * adlen + threadIdx.y;
	uint32_t tcix = blockDim.x * blockIdx.x * (*clen) + threadIdx.x;				//for cipher text
	uint32_t tciy = blockDim.x * blockIdx.x * (*clen) + threadIdx.y;

	//read in col , write in row
	uint32_t tki = tkiy * CRYPTO_KEYBYTES + tkix; // access in rows - key & nonce
	uint32_t tko = tkix * CRYPTO_KEYBYTES + tkiy; // access in columns - key & nonce
	uint32_t tni = tniy * CRYPTO_NPUBBYTES + tnix; // access in rows - key & nonce
	uint32_t tno = tnix * CRYPTO_NPUBBYTES + tniy; // access in columns - key & nonce
	uint32_t tmi = tmiy * mlen + tmix; // access in rows - message 
	uint32_t tmo = tmix * mlen + tmiy; // access in columns - message 
	uint32_t tai = taiy * adlen + taix; // access in columns - ad 
	uint32_t tao = taix * adlen + taiy; // access in columns - ad 
	uint32_t tci = tciy * (*clen) + tcix; // access in row  - cipher

	//temporarily buffer
	uint8_t* kout = const_cast<uint8_t*>(k) + blockIdx.x * blockDim.x;
	uint8_t* nout = const_cast<uint8_t*>(npub) + blockIdx.x * blockDim.x;
	uint8_t* mout = const_cast<uint8_t*>(m) + blockIdx.x * blockDim.x;
	uint8_t* aout = const_cast<uint8_t*>(ad) + blockIdx.x * blockDim.x;

	kout[tko] = k[tki]; // transpose from row to col for key
	nout[tno] = npub[tni]; //for nonce
	mout[tmo] = m[tmi]; //for message
	aout[tao] = ad[tai]; //for additional data

	__syncthreads();

	uint8_t* C = c + tci;
	const uint8_t* M = m + tmo;
	const uint8_t* A = ad + tao;
	const uint8_t* N = npub + tko;
	const uint8_t* K = k + tko;

	Xoodyak_Instance    instance;

	Xoodyak_InitializeG_Op(&instance, K, CRYPTO_KEYBYTES, N, CRYPTO_NPUBBYTES, NULL, 0);
	Xoodyak_AbsorbG_Op(&instance, A, (size_t)adlen);
	Xoodyak_EncryptG_Op(&instance, M, C, (size_t)mlen);
	Xoodyak_SqueezeAnyG_Op(&instance, C + mlen, CRYPTO_ABYTES, 0x40);
	*clen = mlen + CRYPTO_ABYTES;
}

__global__ void crypto_aead_encrypt_gpu_global__Fine(
	uint8_t* c, uint64_t* clen,
	const uint8_t* m, uint64_t mlen,
	const uint8_t* ad, uint64_t adlen,
	const uint8_t* nsec, const uint8_t* npub, const uint8_t* k)
{
	Xoodyak_Instance    instance;

	(void)nsec;

	int tid = threadIdx.x, bid = blockIdx.x;
	uint8_t* C = c + (bid * blockDim.x * (*clen) + ((tid / fineLevel) * (*clen)));
	uint8_t* T = c + mlen + (bid * blockDim.x * (*clen) + ((tid / fineLevel) * (*clen)));
	const uint8_t* M = m + (bid * blockDim.x * mlen + ((tid / fineLevel) * mlen));
	const uint8_t* A = ad + (bid * blockDim.x * adlen + ((tid / fineLevel) * adlen));
	const uint8_t* N = npub + (bid * blockDim.x * CRYPTO_NPUBBYTES + ((tid / fineLevel) * CRYPTO_NPUBBYTES));
	const uint8_t* K = k + (bid * blockDim.x * CRYPTO_KEYBYTES + ((tid / fineLevel) * CRYPTO_KEYBYTES));

	Xoodyak_InitializeG__Fine(&instance, K, CRYPTO_KEYBYTES, N, CRYPTO_NPUBBYTES, NULL, 0);
	Xoodyak_AbsorbG__Fine(&instance, A, (size_t)adlen);
	Xoodyak_EncryptG__Fine(&instance, M, C, (size_t)mlen);
	Xoodyak_SqueezeAnyG__Fine(&instance, C + mlen, CRYPTO_ABYTES, 0x40);
	*clen = mlen + CRYPTO_ABYTES;

}

int main()
{
#ifdef WRITEFILE
	FILE* fpt;
	fpt = fopen("Xoodyak_Concurent_raw.csv", "w");
	fprintf(fpt, "Version, Dimension, Threads, Latency, Memcpy H2D, Transpose, Execution Time, Memcpy D2H, AEAD/s (full latency), AEAD/s (exclude transpose)\n");
#endif

	printf("\nSize Implementation : %d\n", BATCH);

	uint8_t* nonce, * key, * msg, * ad, * ct, * msg2;
	uint64_t alen = ALEN;	// additional data length
	uint64_t mlen = MLEN;	// messege length
	uint64_t clen;	// cipher length
	LARGE_INTEGER frequency;
	LARGE_INTEGER t1, t2;
	double cpu_t = 0;

	cudaMallocHost((void**)&key, BATCH * CRYPTO_KEYBYTES * sizeof(uint8_t));
	cudaMallocHost((void**)&nonce, BATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t));
	cudaMallocHost((void**)&msg, BATCH * mlen * sizeof(uint8_t));
	cudaMallocHost((void**)&ad, BATCH * alen * sizeof(uint8_t));
	cudaMallocHost((void**)&ct, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));

	init_buffer('k', key, CRYPTO_KEYBYTES);
	init_buffer('n', nonce, CRYPTO_NPUBBYTES);
	init_buffer('m', msg, mlen);
	init_buffer('a', ad, alen);

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&t1);
	for (int i = 0; i < BATCH; i++) {

#ifdef PRINT
		print('k', key + (i * CRYPTO_KEYBYTES), CRYPTO_KEYBYTES);
		printf(" ");
		print('n', nonce + (i * CRYPTO_NPUBBYTES), CRYPTO_NPUBBYTES);
		print('a', ad + (i * alen), alen);
		printf(" ");
		print('m', msg + (i * mlen), mlen);
		printf(" -> ");
#endif

		int result = crypto_aead_encrypt(OFFSET(ct, i, clen), &clen, OFFSET(msg, i, mlen), mlen, OFFSET(ad, i, alen),
			alen, NULL, OFFSET(nonce, i, CRYPTO_NPUBBYTES), OFFSET(key, i, CRYPTO_KEYBYTES));

#ifdef PRINTC
		print('c', ct + (i * clen), clen);
#endif
	}

	QueryPerformanceCounter(&t2);
	cpu_t += ((double)(t2.QuadPart - t1.QuadPart) * 1000.0 / (double)frequency.QuadPart);

	//Print Time
	printf("Version\t\tCKernel\tConfiguration\tMemcpyH2D\tMemcpyD2H\tLatency\t\tAEAD/s (full latency)\t AEAD/s (exclude transpose)\n\n");
	printf("Host\t\t-\t\t-\t\t\t%.6f\t%.6f\t%.6f \t( %.2f )\t%.6f  ( %.2f )\t%.f\n", 0.0, 0.0, cpu_t, 0.0, cpu_t, 0.0, BATCH / (cpu_t / 1000));
#ifdef WRITEFILE
	fprintf(fpt, "%s, %d, %d, %.6f, %.6f, %.6f, %.6f,%.6f, %.6f, %.6f, %.6f, %.f, %.2f\n", "Host", 0, 0, ((BATCH * clen * sizeof(uint8_t)) * 1e-6) / cpu_t, cpu_t, 0.0, (((BATCH * clen * sizeof(uint8_t)) * 1e-6) / cpu_t) * 8, 0.0, cpu_t, 0.0, 0.0, BATCH / (cpu_t / 1000), 0.0);
#endif

	//GPU implementation
	uint8_t* d_n, * d_k, * d_a, * d_m, * d_c, * h_c;
	uint64_t* d_clen;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Memory Allocation - Device
	cudaMallocHost((void**)&h_c, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));		//Host Cipher
	cudaMalloc((void**)&d_c, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));			//Device Cipher
	cudaMalloc((void**)&d_n, BATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t));			//Nonce
	cudaMalloc((void**)&d_k, BATCH * CRYPTO_KEYBYTES * sizeof(uint8_t));				//Key
	cudaMalloc((void**)&d_m, BATCH * (uint64_t)mlen * sizeof(uint8_t));				//Message
	cudaMalloc((void**)&d_a, BATCH * (uint64_t)alen * sizeof(uint8_t));				//Additional Data
	cudaMallocHost((void**)&d_clen, sizeof(uint64_t));

	//Memory initialisation
	memset(h_c, 0, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
	cudaMemset(d_c, 0, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
	cudaMemset(d_n, 0, BATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t));
	cudaMemset(d_k, 0, BATCH * CRYPTO_KEYBYTES * sizeof(uint8_t));
	cudaMemset(d_m, 0, BATCH * (uint64_t)mlen * sizeof(uint8_t));
	cudaMemset(d_a, 0, BATCH * (uint64_t)alen * sizeof(uint8_t));

	//Warm Up Kernel
	cudaEventRecord(start, 0);
	crypto_aead_encrypt_gpu_global << <BATCH / 1, 1 >> > (d_c, d_clen, d_m, mlen, d_a, alen, NULL, d_n, d_k);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float warmup;
	cudaEventElapsedTime(&warmup, start, stop);

	void (*kernel)(uint8_t*, uint64_t*, const uint8_t*, uint64_t, const uint8_t*, uint64_t, const uint8_t*, const uint8_t*, const uint8_t*);
	size_t size = BATCH * (*d_clen) * sizeof(uint8_t);

	cudaStream_t GPUs2[2], GPUs4[4], GPUs5[5];
	cudaStream_t* GPUstreams;

	for (int z = 2; z <= NSTREAM_SIZE; z++) {
		if (z != 3) {
			switch (z) {
			case 2: {GPUstreams = GPUs2; break; }
			case 4: {GPUstreams = GPUs4; break; }
			case 5: {GPUstreams = GPUs5; break; }
			}

			for (int a = 0; a < z; a++) {	//1 streams 8 bits
				CHECK(cudaStreamCreate(&GPUstreams[a]));
			}

			//Determine data size
			int iBATCH = BATCH / z;
			size_t iKeysize = iBATCH * CRYPTO_KEYBYTES * sizeof(uint8_t);
			size_t iNoncesize = iBATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t);
			size_t iMsgsize = iBATCH * (uint64_t)mlen * sizeof(uint8_t);
			size_t iAdsize = iBATCH * (uint64_t)alen * sizeof(uint8_t);
			size_t iCsize = iBATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t);

			cudaEventRecord(start, 0);
			for (int i = 0; i < z; ++i)
			{
				int ioffset = i * iBATCH;
				cudaMemcpyAsync(&d_n[ioffset * CRYPTO_NPUBBYTES], &nonce[ioffset * CRYPTO_NPUBBYTES], iNoncesize, cudaMemcpyHostToDevice, GPUstreams[i]);
				cudaMemcpyAsync(&d_k[ioffset * CRYPTO_KEYBYTES], &key[ioffset * CRYPTO_KEYBYTES], iKeysize, cudaMemcpyHostToDevice, GPUstreams[i]);
				cudaMemcpyAsync(&d_m[ioffset * mlen], &msg[ioffset * mlen], iMsgsize, cudaMemcpyHostToDevice, GPUstreams[i]);
				cudaMemcpyAsync(&d_a[ioffset * alen], &ad[ioffset * alen], iAdsize, cudaMemcpyHostToDevice, GPUstreams[i]);
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float memcpy_h2d;
			cudaEventElapsedTime(&memcpy_h2d, start, stop);

			for (int i = 64; i < 513; i *= 2) {

				float elapsed, memcpy_d2h, total;

				for (int a = 1; a <= 5; a++) {

					//Configuration.
					dim3 threads(i);
					double temp = (double)iBATCH / (double)i;
					dim3 blocks(ceil(temp));		//for unoptimised

					if (a > 1) {
						threads.y = i;
						temp = (double)iBATCH / ((double)threads.x * (double)threads.y);
						blocks.x = ceil(temp);
						blocks.x = (blocks.x < 1) ? 1 : blocks.x; // at least 1 block
					}

					kernel = ((a == 1) ? &crypto_aead_encrypt_gpu_global : ((a == 2) ? &crypto_aead_encrypt_gpu_global_Op :
						(((a == 3) ? &crypto_aead_encrypt_gpu_rcwr_GpuTranspose : ((a == 4) ? &crypto_aead_encrypt_gpu_rcwr_GPUTransposeUnroll4
							: &crypto_aead_encrypt_gpu_rcwr_GpuTranspose_Op)))));
					char* kernelName = ((a == 1) ? "GPU Unoptimised" : ((a == 2) ? "GPU Op" : (((a == 3) ? "GPU Tran" :
						((a == 4) ? "GPU TransU4" : "GPU Op Trans")))));

					//Kernel execution
					memset(h_c, 0, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
					cudaEventRecord(start);
					for (int i = 0; i < z; ++i) {
						int ioffset = i * iBATCH;
						kernel << <blocks, threads, 0, GPUstreams[i] >> > (&d_c[ioffset * MAX_CIPHER_LENGTH], d_clen, &d_m[ioffset * mlen], mlen, &d_a[ioffset * alen], alen, 0,
							&d_n[ioffset * CRYPTO_NPUBBYTES], &d_k[ioffset * CRYPTO_KEYBYTES]);
					}
					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					elapsed = 0;
					cudaEventElapsedTime(&elapsed, start, stop);

					//Memory Copy from D2H
					cudaEventRecord(start, 0);
					for (int i = 0; i < z; ++i) {
						int ioffset = i * iBATCH;
						cudaMemcpyAsync(&h_c[ioffset * MAX_CIPHER_LENGTH], &d_c[ioffset * MAX_CIPHER_LENGTH], iCsize, cudaMemcpyDeviceToHost, GPUstreams[i]);
					}
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					memcpy_d2h = 0;
					cudaEventElapsedTime(&memcpy_d2h, start, stop);

					checkResult(kernelName, ct, h_c, MAX_CIPHER_LENGTH);

					total = memcpy_h2d + elapsed + memcpy_d2h;

					printf("%s\t%d\t %u \t\t%.6f\t%.6f\t%.6f  \t%.f \t\t%.f\n", kernelName, z, threads.x, memcpy_h2d,
						memcpy_d2h, total, BATCH / (total / 1000), BATCH / ((total) / 1000));

#ifdef WRITEFILE
					fprintf(fpt, "%s,%d, %u, %.6f, %.6f, %.6f, %.6f,  %.6f, %.f, %.f\n", kernelName, z, threads.x, total,
						memcpy_h2d, Ttime, elapsed, memcpy_d2h, BATCH / (total / 1000), BATCH / ((total) / 1000));
#endif
				}
				printf("\n");
			}

			//Fine grain
			size_t size = BATCH * (clen) * sizeof(uint8_t);
			dim3 threads2(Tlimit); //fine grain each block max 512 threads to divide by 4/8/16 threads for fine grain.
			double temp = ((double)BATCH / (Tlimit / (double)fineLevel));
			dim3 blocks2(ceil(temp));		//for unoptimised
			size_t iSsize = iBATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t);

			memset(h_c, 0, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
			cudaEventRecord(start);
			for (int i = 0; i < z; ++i) {
				int ioffset = i * iBATCH;
				crypto_aead_encrypt_gpu_global__Fine << <blocks2, threads2, 0, GPUstreams[i] >> > (&d_c[ioffset * MAX_CIPHER_LENGTH], d_clen, &d_m[ioffset * mlen], mlen, &d_a[ioffset * alen], alen, 0,
					&d_n[ioffset * CRYPTO_NPUBBYTES], &d_k[ioffset * CRYPTO_KEYBYTES]);
			}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float elapsed = 0;
			cudaEventElapsedTime(&elapsed, start, stop);

			//Memory Copy from D2H
			cudaEventRecord(start, 0);
			for (int i = 0; i < z; ++i) {
				int ioffset = i * iBATCH;
				cudaMemcpyAsync(&h_c[ioffset * MAX_CIPHER_LENGTH], &d_c[ioffset * MAX_CIPHER_LENGTH], iCsize, cudaMemcpyDeviceToHost, GPUstreams[i]);
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float memcpy_d2h = 0;
			cudaEventElapsedTime(&memcpy_d2h, start, stop);

			checkResult("Fine Grain ", ct, h_c, MAX_CIPHER_LENGTH);
			float total = memcpy_h2d + elapsed + memcpy_d2h;

			printf("%s%d\t%d\t\t%.6f\t%.6f\t%.6f\t%.f\n", "Fine ", fineLevel, z, memcpy_h2d,
				memcpy_d2h, total, BATCH / (total / 1000));

#ifdef WRITEFILE
			fprintf(fpt, "%s%d, %d, (%u), %.6f, %.6f, %.6f, %.6f,%.6f, %.6f, %.6f, %.6f, %.f,%.2f\n", "Fine ", fineLevel, z,  threads2.x, (size * 2e-6) / total, total,
				cpu_t / total, (size * 2e-6) / total * 8, memcpy_h2d, elapsed, memcpy_d2h, cpu_t / elapsed, BATCH / (total / 1000), (BATCH / (total / 1000)) / (BATCH / (cpu_t / 1000)));
# endif

			printf("\n======================================================================================================================================================\n");
			for (int i = 0; i < z; i++)
				cudaStreamDestroy(GPUstreams[i]);
		}


	}

	//Free Memory
	//Host memory
	cudaFree(nonce);
	cudaFree(key);
	cudaFree(msg);
	cudaFree(ad);
	cudaFree(ct);

	//Device memory
	cudaFree(d_n);
	cudaFree(d_k);
	cudaFree(d_a);
	cudaFree(d_m);
	cudaFree(d_c);
	cudaFree(h_c);
	cudaFree(d_clen);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#ifdef WRITEFILE
	fclose(fpt);
#endif
	cudaDeviceReset();

	return 0;
}