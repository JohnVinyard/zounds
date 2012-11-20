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
// Parameter ##################################################################
typedef struct {
	// the current value of a parameter is computed using the current time,
	// the previous value, the next value, and the interpolation type.

	// a list of parameter values
	float * values;
	int n_values;

	// a list of times at which the parameter values should occur. This must
	// be len(values) - 1, since the first value is assumed to begin at time zero.
	jack_nframes_t * times;

	// the type of interpolation for the current transition. This must be
	// len(values) - 1, , since the first value is assumed to begin at time zero.
	// Implement the interpolation types recommended by  the Web Audio API
	int * interpolation;

	// the position in the values array
	int pos;

} parameter;

// Create a new parameter instance
parameter parameter_new(float * values,         // all values for this parameter
						int n_values,           // the number of values
						jack_nframes_t * times, // times at which the values begin
						int * interpolations);   // interpolation type codes

// Get the current value of the parameter instance
void parameter_current_value(parameter * param,           // parameter instance
							jack_nframes_t time,          // current time
							float previous_value,         // previous param value
							float next_value,             // next param value
							jack_nframes_t previous_time, // begin interpolate time
							jack_nframes_t next_time);    // end interpolate time

// Advance the parameter to the next transition if the next way-point has been
// reached.
void parameter_advance_if_necessary(parameter * param,
									jack_nframes_t current_time);


// Transform ##################################################################
typedef struct {
	// parameter list for this transform. the process() function for each transform
	// type is responsible for knowing what order the parameters should arrive in.
	struct parameter * parameters;
	int n_parameters;

	// a float array containing any state needed by this transform. Note that
	// this should be allocated at scheduling time, outside of the realtime
	// thread.
	float * state_buf;
	int state_buf_size;
	int state_buf_pos;

	// TODO: What about non-param values, like the filter type, or "normalize"
	// for the convolution effect?

	// a function with the signature
	// process(in_buffer,out_buffer,struct parameter *parameters,float * state_buf)
	// E.g., For a delay, this will contain the "hold" buffer, for a convolution, the
	// impulse response
	void * process;

} transform;

// Construct a new transform
// TODO: How do I handle transform-specific constructor parameters, like filter
// type, or delay time, e.g.?
transform transform_new(parameter * params,
						int n_parameters,
						int state_buf_size,
						void * process);

// Event ######################################################################
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
	// defaults to 0
	int n_children;


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

