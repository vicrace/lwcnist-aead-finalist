﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include "params.h"
#include "operations.h"
#include "photonBeetle.h"

static int crypto_aead_encrypt(
	const uint8_t* key,   // 16 -bytes secret key
	const uint8_t* nonce, // 16 -bytes public message nonce
	const uint8_t* data,  // N -bytes associated data | N >= 0
	const size_t dlen,                     // len(data) >= 0
	const uint8_t* txt,   // N -bytes plain text | N >= 0
	uint8_t* enc,         // N -bytes cipher text | N >= 0
	const size_t mlen,                     // len(txt) = len(enc) >= 0
	uint8_t* tag          // 16 -bytes authentication tag
)
{
	if (check_rate(R)) {

		uint8_t* C = enc;
		uint8_t* T = tag;
		const uint8_t* M = txt;
		const uint8_t* A = data;
		const uint8_t* N = nonce;
		const uint8_t* K = key;

		uint8_t state[32];

		memcpy(state, N, NONCE_LEN);
		memcpy(state + NONCE_LEN, K, KEY_LEN);

		if ((dlen == 0) && (mlen == 0)) {
			state[31] ^= (1 << 5);
			gen_tag(state, T, TAG_LEN);

			return 0;
		}

		const bool f0 = mlen > 0;
		const bool f1 = (dlen & (R - 1)) == 0;
		const bool f2 = dlen > 0;
		const bool f3 = (mlen & (R - 1)) == 0;

		const uint8_t C0 = (f0 && f1) ? 1 : f0 ? 2 : f1 ? 3 : 4;
		const uint8_t C1 = (f2 && f3) ? 1 : f2 ? 2 : f3 ? 5 : 6;

		if (dlen > 0) {
			absorb(state, A, dlen, C0, R);
		}

		if (mlen > 0) {
			for (size_t off = 0; off < mlen; off += R) {
				photon256(state);
				const auto len = ((R < (mlen - off)) ? R : (mlen - off));
				rho(state, M + off, C + off, len);
			}

			state[31] ^= (C1 << 5);
		}

		gen_tag(state, T, TAG_LEN);
		return 0;
	}
	return -1;
}

__global__ void crypto_aead_encrypt_gpu_global(
	const uint8_t * key,   // 16 -bytes secret key
	const uint8_t * nonce, // 16 -bytes public message nonce
	const uint8_t * data,  // N -bytes associated data | N >= 0
	const size_t dlen,                     // len(data) >= 0
	const uint8_t * txt,   // N -bytes plain text | N >= 0
	uint8_t * enc,         // N -bytes cipher text | N >= 0
	const size_t mlen,                     // len(txt) = len(enc) >= 0
	uint8_t * tag          // 16 -bytes authentication tag
) {
	if (check_rateG(R)) {

		int tid = threadIdx.x, bid = blockIdx.x;
		uint32_t idx_im = bid * blockDim.x * mlen + tid * mlen;
		uint32_t idx_ia = bid * blockDim.x * dlen + tid * dlen;			// AD
		uint32_t idx_in = bid * blockDim.x * CRYPTO_NPUBBYTES + tid * CRYPTO_NPUBBYTES; //key and nonce read only 16
		uint32_t idx_ik = bid * blockDim.x * CRYPTO_KEYBYTES + tid * CRYPTO_KEYBYTES; //key and nonce read only 16
		uint32_t idx_out = bid * blockDim.x * MAX_CIPHER_LENGTH + (tid * MAX_CIPHER_LENGTH);	//instead of crypto_abytes
		uint32_t idx_tag = bid * blockDim.x * TAG_LEN + (tid * TAG_LEN);	//instead of crypto_abytes

		uint8_t* C = enc + idx_out;
		uint8_t* T = tag + idx_tag;
		const uint8_t* M = txt + idx_im;
		const uint8_t* A = data + idx_ia;
		const uint8_t* N = nonce + idx_in;
		const uint8_t* K = key + idx_ik;

		uint8_t state[32];

		memcpy(state, N, NONCE_LENG);
		memcpy(state + NONCE_LENG, K, KEY_LENG);

		if ((dlen == 0) && (mlen == 0)) [[unlikely]] {
		  state[31] ^= (1 << 5);
		  gen_tagG(state, T, TAG_LENG);
		}

		const bool f0 = mlen > 0;
		const bool f1 = (dlen & (R - 1)) == 0;
		const bool f2 = dlen > 0;
		const bool f3 = (mlen & (R - 1)) == 0;

		const uint8_t C0 = (f0 && f1) ? 1 : f0 ? 2 : f1 ? 3 : 4;
		const uint8_t C1 = (f2 && f3) ? 1 : f2 ? 2 : f3 ? 5 : 6;

		if (dlen > 0) [[likely]] {
		  absorbG(state, A, dlen, C0,R);
		}

			if (mlen > 0) [[likely]] {
			  for (size_t off = 0; off < mlen; off += R) {
				photon256G(state);
				const auto len = ((R < (mlen - off)) ? R : (mlen - off));
				rhoG(state, M + off, C + off, len);
			  }

			  state[31] ^= (C1 << 5);
			}

		gen_tagG(state, T, TAG_LENG);
	}
}


__global__ void crypto_aead_encrypt_gpu_global_Op(
	const uint8_t * key,   // 16 -bytes secret key
	const uint8_t * nonce, // 16 -bytes public message nonce
	const uint8_t * data,  // N -bytes associated data | N >= 0
	const size_t dlen,                     // len(data) >= 0
	const uint8_t * txt,   // N -bytes plain text | N >= 0
	uint8_t * enc,         // N -bytes cipher text | N >= 0
	const size_t mlen,                     // len(txt) = len(enc) >= 0
	uint8_t * tag          // 16 -bytes authentication tag
) {
	if (check_rateG(R)) {

		int tid = threadIdx.x, bid = blockIdx.x;
		uint32_t idx_im = bid * blockDim.x * mlen + tid * mlen;
		uint32_t idx_ia = bid * blockDim.x * dlen + tid * dlen;			// AD
		uint32_t idx_in = bid * blockDim.x * CRYPTO_NPUBBYTES + tid * CRYPTO_NPUBBYTES; //key and nonce read only 16
		uint32_t idx_ik = bid * blockDim.x * CRYPTO_KEYBYTES + tid * CRYPTO_KEYBYTES; //key and nonce read only 16
		uint32_t idx_out = bid * blockDim.x * MAX_CIPHER_LENGTH + (tid * MAX_CIPHER_LENGTH);	//instead of crypto_abytes
		uint32_t idx_tag = bid * blockDim.x * TAG_LEN + (tid * TAG_LEN);	//instead of crypto_abytes

		uint8_t* C = enc + idx_out;
		uint8_t* T = tag + idx_tag;
		const uint8_t* M = txt + idx_im;
		const uint8_t* A = data + idx_ia;
		const uint8_t* N = nonce + idx_in;
		const uint8_t* K = key + idx_ik;

		uint8_t state[32];

		memcpy(state, N, NONCE_LENG);
		memcpy(state + NONCE_LENG, K, KEY_LENG);

		if ((dlen == 0) && (mlen == 0)) [[unlikely]] {
			state[31] ^= (1 << 5);
			gen_tagG_Op(state, T, TAG_LENG);
		}

		const bool f0 = mlen > 0;
		const bool f1 = (dlen & (R - 1)) == 0;
		const bool f2 = dlen > 0;
		const bool f3 = (mlen & (R - 1)) == 0;

		const uint8_t C0 = (f0 && f1) ? 1 : f0 ? 2 : f1 ? 3 : 4;
		const uint8_t C1 = (f2 && f3) ? 1 : f2 ? 2 : f3 ? 5 : 6;

		if (dlen > 0) [[likely]] {
		  absorbG_Op(state, A, dlen, C0,R);
		}

			if (mlen > 0) [[likely]] {
			  for (size_t off = 0; off < mlen; off += R) {
				photon256G_Op(state);
				const auto len = ((R < (mlen - off)) ? R : (mlen - off));
				rhoG(state, M + off, C + off, len);
			  }

			  state[31] ^= (C1 << 5);
			}

		gen_tagG_Op(state, T, TAG_LENG);
	}
}


int main()
{
#ifdef WRITEFILE
	//FILE writing
	FILE* fpt;
	fpt = fopen("Photon8x4_CK.csv", "w");
	fprintf(fpt, "Version, Dimension, Threads, Latency, Memcpy H2D, Transpose, Execution Time, Memcpy D2H, AEAD/s (full latency), AEAD/s (exclude transpose)\n");
#endif

	printf("\nSize Implementation : %d\n", BATCH);

	//Host variable
	uint8_t* nonce, * key, * msg, * ad, * ct, * tag;
	uint64_t alen = ALEN;	// additional data length
	uint64_t mlen = MLEN;	// messege length
	uint64_t clen = MLEN;	// cipher length
	LARGE_INTEGER frequency;
	LARGE_INTEGER t1, t2;
	double cpu_t = 0;

	//Memory Allocation - HOST
	cudaMallocHost((void**)& key, BATCH * CRYPTO_KEYBYTES * sizeof(uint8_t));
	cudaMallocHost((void**)& nonce, BATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t));
	cudaMallocHost((void**)& msg, BATCH * mlen * sizeof(uint8_t));
	cudaMallocHost((void**)& ad, BATCH * alen * sizeof(uint8_t));
	cudaMallocHost((void**)& ct, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
	cudaMallocHost((void**)& tag, BATCH * TAG_LEN * sizeof(uint8_t));

	init_buffer('k', key, CRYPTO_KEYBYTES);
	init_buffer('n', nonce, CRYPTO_NPUBBYTES);
	init_buffer('m', msg, mlen);
	init_buffer('a', ad, alen);

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&t1);
	for (int i = 0; i < BATCH; i++) {

		int result = crypto_aead_encrypt(OFFSET(key, i, CRYPTO_KEYBYTES), OFFSET(nonce, i, CRYPTO_NPUBBYTES), OFFSET(ad, i, alen),
			MAX_ASSOCIATED_DATA_LENGTH, OFFSET(msg, i, mlen), OFFSET(ct, i, clen), MAX_MESSAGE_LENGTH, OFFSET(tag, i, TAG_LEN));
	}
	QueryPerformanceCounter(&t2);
	cpu_t += ((double)(t2.QuadPart - t1.QuadPart) * 1000.0 / (double)frequency.QuadPart);
	//Print Host Time
	printf("Version\t\tCKernel\tConfiguration\tMemcpyH2D\tMemcpyD2H\tLatency\t\tAEAD/s\t AEAD/s (exclude transpose)\n\n");
#ifdef WRITEFILE
	fprintf(fpt, "%s, %.6f, %.6f, %.6f,%.6f, %.6f, %.6f, %.6f, %.6f, %.f, %.2f\n", "Host ", 0, 0.0, cpu_t, 0.0, 0.0, cpu_t, 0.0, BATCH / (cpu_t / 1000), BATCH / (cpu_t / 1000));
#endif
	printf("Host \t\tSerial\t\t\t%.6f\t%.6f\t%.6f\t%.f\t%.f\n", 0.0, 0.0, cpu_t, BATCH / (cpu_t / 1000), BATCH / (cpu_t / 1000));

	//GPU implementation
	LARGE_INTEGER frequencyT;
	LARGE_INTEGER TS, TE;
	uint8_t * key_out, *msg_out, *ad_out, *nonce_out;

	cudaMallocHost((void**)& key_out, BATCH * CRYPTO_KEYBYTES * sizeof(uint8_t));
	cudaMallocHost((void**)& msg_out, BATCH * mlen * sizeof(uint8_t));
	cudaMallocHost((void**)& ad_out, BATCH * alen * sizeof(uint8_t));
	cudaMallocHost((void**)& nonce_out, BATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t));

	uint8_t * d_n, *d_k, *d_a, *d_m, *d_c, *h_c, *h_t, *d_t;
	uint64_t * d_clen;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Memory Allocation - Device
	cudaMallocHost((void**)& h_c, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));		//Host Cipher
	cudaMalloc((void**)& d_c, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));			//Device Cipher
	cudaMalloc((void**)& d_n, BATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t));			//Nonce
	cudaMalloc((void**)& d_k, BATCH * CRYPTO_KEYBYTES * sizeof(uint8_t));				//Key
	cudaMalloc((void**)& d_m, BATCH * (uint64_t)mlen * sizeof(uint8_t));				//Message
	cudaMalloc((void**)& d_a, BATCH * (uint64_t)alen * sizeof(uint8_t));				//Additional Data
	cudaMalloc((void**)& d_t, BATCH * (uint64_t)TAG_LEN * sizeof(uint8_t));				//Message
	cudaMalloc((void**)& h_t, BATCH * (uint64_t)TAG_LEN * sizeof(uint8_t));				//Additional Data
	cudaMallocHost((void**)& d_clen, sizeof(uint64_t));

	//Memory initialisation
	memset(h_c, 0, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
	cudaMemset(d_c, 0, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
	cudaMemset(d_n, 0, BATCH * CRYPTO_NPUBBYTES * sizeof(uint8_t));
	cudaMemset(d_k, 0, BATCH * CRYPTO_KEYBYTES * sizeof(uint8_t));
	cudaMemset(d_m, 0, BATCH * (uint64_t)mlen * sizeof(uint8_t));
	cudaMemset(d_a, 0, BATCH * (uint64_t)alen * sizeof(uint8_t));
	cudaMemset(d_t, 0, BATCH * (uint64_t)TAG_LEN * sizeof(uint8_t));
	cudaMemset(d_t, 0, BATCH * (uint64_t)TAG_LEN * sizeof(uint8_t));
	*d_clen = MAX_CIPHER_LENGTH;

	void (*kernel)(const uint8_t*, const uint8_t*, const uint8_t*, size_t, const uint8_t*, uint8_t*, const size_t, uint8_t*);
	size_t size = BATCH * (*d_clen) * sizeof(uint8_t);

	cudaStream_t GPUs2[2], GPUs4[4], GPUs5[5];
	cudaStream_t * GPUstreams;

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
			size_t iTsize = iBATCH * TAG_LEN * sizeof(uint8_t);

			for (int i = 64; i < 1025; i *= 2) {

				float memcpy_h2d, elapsed, memcpy_d2h, total;
				for (int a = 1; a <3; a++) {


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
					memcpy_h2d = 0.0f;
					cudaEventElapsedTime(&memcpy_h2d, start, stop);


					//Configuration.
					dim3 threads(i);
					double temp = (double)iBATCH / (double)i;
					dim3 blocks(ceil(temp));


					kernel = ((a == 1) ? &crypto_aead_encrypt_gpu_global : &crypto_aead_encrypt_gpu_global_Op);
					char* kernelName = ((a == 1) ? "GPU Ref      " : "GPU Op  ");

					//Kernel execution
					memset(h_c, 0, BATCH * MAX_CIPHER_LENGTH * sizeof(uint8_t));
					cudaEventRecord(start);
					for (int i = 0; i < z; ++i) {
						int ioffset = i * iBATCH;
						kernel << <blocks, threads, 0, GPUstreams[i] >> > (&d_k[ioffset * CRYPTO_KEYBYTES], &d_n[ioffset * CRYPTO_NPUBBYTES], &d_a[ioffset * alen],
							MAX_ASSOCIATED_DATA_LENGTH, &d_m[ioffset * mlen], &d_c[ioffset * MAX_CIPHER_LENGTH], MAX_MESSAGE_LENGTH, &d_t[ioffset * TAG_LEN]);
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
						cudaMemcpyAsync(&h_t[ioffset * TAG_LEN], &d_c[ioffset * TAG_LEN], iTsize, cudaMemcpyDeviceToHost, GPUstreams[i]);
					}
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					memcpy_d2h = 0;
					cudaEventElapsedTime(&memcpy_d2h, start, stop);
					checkResult(kernelName, ct, h_c, MAX_CIPHER_LENGTH);

					double Ttime = 0;

					total = memcpy_h2d + elapsed + memcpy_d2h;


					printf("%s\t %d\t %u \t\t%.6f\t%.6f \t%.6f  \t%.f   \t%.f\n", kernelName, z, threads.x, memcpy_h2d,
						memcpy_d2h, total, BATCH / (total / 1000), BATCH / ((total - Ttime) / 1000));
#ifdef WRITEFILE
					fprintf(fpt, "%s,%d, %u, %.6f, %.6f, %.6f, %.6f,  %.6f, %.f, %.f\n", kernelName, z, threads.x, total,
						memcpy_h2d, Ttime, elapsed, memcpy_d2h, BATCH / (total / 1000), BATCH / ((total - Ttime) / 1000));
#endif
				}
				printf("\n");
			}
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
	cudaFree(tag);

	//Device memory
	cudaFree(d_n);
	cudaFree(d_k);
	cudaFree(d_a);
	cudaFree(d_m);
	cudaFree(d_c);
	cudaFree(h_c);
	cudaFree(h_t);
	cudaFree(d_t);
	cudaFree(d_clen);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

#ifdef WRITEFILE
	fclose(fpt);
#endif
	cudaDeviceReset();
	return 0;
}
