#include <jack/jack.h>

#define CHANNELS 2


typedef int (*t_callback)(int, float *, float **);

jack_client_t *jack_start(void);
void setup(void);
void teardown(void);
jack_time_t get_time(void);

typedef struct {
	// a buffer containing audio samples
	float * buf;
	// the position in the buffer to start
	unsigned int start_sample;
	// the stop position in the buffer
	unsigned int stop_sample;
	// the start time, in microseconds
	jack_nframes_t start_time_frames;
	// a flag indicating that all samples have been output
	char done;
	// the current position in the sample
	unsigned int position;

} event_t;

#define N_EVENTS 256
#define second_in_microsecs 1e6

void put_event(
float *buf,unsigned int start_sample,unsigned int stop_sample,jack_time_t start_time_ms,char done);
void cancel_all_events(void);

