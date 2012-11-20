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

/*
typedef struct {
	// the current value of a parameter is computed using the current time,
	// the previous value, the next value, and the interpolation type.

	// a list of parameter values
	float * values;
	// a list of times at which the parameter values should occur. This must
	// be len(values) - 1
	jack_nframes_t * times;
	// the position in the values array
	int pos;
	// the type of interpolation for the current transition. This must be
	// len(values) - 1
	int * interpolation;
} parameter;



typedef struct {
	// ## LEAF PATTERN DATA ###############################################

	// a buffer containing audio samples. This is a leaf node.
	float * buf;
	// the position in the buffer to start
	unsigned int start_sample;
	// the stop position in the buffer
	unsigned int stop_sample;
	// the current position in the sample
	unsigned int position;

	// ## BRANCH PATTERN DATA ###############################################

	// an array of child events. this should be NULL if this is a "leaf" event
	struct event2_t * children;


	// ## COMMON DATA ######################################################

	// the start time, in microseconds
	jack_nframes_t start_time_frames;
	// a flag indicating that all samples have been output
	char done;
	// a flag indicating that one of the effects defined for this event
	// cause it to have an undetermined length. Note that if *any* "descendant"
	// nodes have an unknown length, then this node should as well
	char unkown_length;
	// a flag for use with events who have unknown length.  This is 1 if
	// the samples have finished playing and the output has been "silent" (tbd)
	// for some tbd length of time
	char silent;

} event2_t;
*/


#define N_EVENTS 256
#define second_in_microsecs 1e6

void put_event(
float *buf,unsigned int start_sample,unsigned int stop_sample,jack_time_t start_time_ms,char done);
void cancel_all_events(void);

