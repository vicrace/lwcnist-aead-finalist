// GPU
#define CRYPTO_KEYBYTES		16
#define CRYPTO_NSECBYTES	0
#define CRYPTO_NPUBBYTES	12
#define CRYPTO_ABYTES		8
#define CRYPTO_NOOVERLAP	1

#define MAX_MESSAGE_LENGTH 64 //32
#define MAX_ASSOCIATED_DATA_LENGTH 0// 32
#define ALEN MAX_ASSOCIATED_DATA_LENGTH
#define MLEN MAX_MESSAGE_LENGTH
#define MAX_CIPHER_LENGTH (MLEN + CRYPTO_ABYTES)
#define OFFSET(arr,i,offset) (arr + (i*offset))			
//#define PRINT
//#define PRINTC
#define BATCH	64000//16000000
#define fineLevel 2 
//#define WRITEFILE

//Ref
#define STREAM_BYTES	16
#define MSG_BYTES		0
