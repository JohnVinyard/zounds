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



// Parameter ##################################################################
enum INTERPOLATION_TYPE {
	Jump,
	Linear,
	Exponential
};

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
	char * interpolations;

	// the position in the values array
	int pos;

} parameter;

// Create a new parameter instance
parameter * parameter_new(float * values,         // all values for this parameter
						int n_values,           // the number of values
						jack_nframes_t * times, // times at which the values begin
						int * interpolations);   // interpolation type codes

// Get the current value of the parameter instance
float parameter_current_value(parameter * param,    // parameter instance
							  jack_nframes_t time); // current time

// Advance the parameter to the next transition if the next way-point has been
// reached.  Return the current position in the values array.
int parameter_advance_if_necessary(parameter * param,
									jack_nframes_t current_time);

// Free the memory used by the parameter
void parameter_delete(parameter * param);


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
transform * transform_new(parameter * params,
						int n_parameters,
						int state_buf_size,
						float (* process)(jack_nframes_t time,float insample,transform * t));

void transform_delete(transform * t);

// Gain
transform * gain_new(float * values, int n_values,jack_nframes_t * times,char * interps);
float gain_process(jack_nrames_t time,float insample,transform * gain);

// Delay
transform * delay_new(int max_delay_time,parameter * params);
float delay_process(jack_nframes_t time,float insample,transform * delay);

// Event ######################################################################
typedef struct {
	// ## LEAF PATTERN DATA ###############################################

	// a buffer containing audio samples. This is a leaf node.
	float * buf;
	// the position in the buffer to start
	unsigned int start_sample;
	// the stop position in the buffer
	unsigned int stop_sample;
	unsigned int n_samples;
	// the current position in the sample
	unsigned int position;

	// ## BRANCH PATTERN DATA ###############################################

	// an array of child events. this should be NULL if this is a "leaf" event
	struct event2 * children;
	// defaults to 0
	int n_children;


	// ## COMMON DATA ######################################################
	// transforms to apply to this event
	transform * transforms;
	int n_transforms;
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

} event2;

event2 * event2_new_leaf(
float * buf,int start_sample,int stop_sample,jack_nframes_t start_time,
char unknown_length,transform * transforms,int n_transforms);

event2 * event2_new_branch(
event2 * children,int n_children,jack_nframes_t start_time,
char unknown_length,transform * transforms,int n_transforms);

char event2_is_leaf(event2 * event);

float event2_process(event2 * event,jack_nframes_t time);

void event2_delete(event2 * event);


#define N_EVENTS 256
#define second_in_microsecs 1e6

void put_event(
float *buf,unsigned int start_sample,unsigned int stop_sample,jack_time_t start_time_ms,char done);
void cancel_all_events(void);

