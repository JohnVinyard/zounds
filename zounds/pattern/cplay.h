#include <jack/jack.h>

#define CHANNELS 2

// KLUDGE: What about other JACK sample rates?
#define KILL_AFTER 44100 * 10
#define SILENCE_THRESHOLD .001

typedef int (*t_callback)(int, float *, float **);

jack_client_t *jack_start(void);
void init_events(void);
void setup(void);
void teardown(void);
jack_time_t get_time(void);
jack_nframes_t get_frame_time(void);

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
void parameter_init(parameter * param,
					float * values,         // all values for this parameter
					int n_values,           // the number of values
					jack_nframes_t * times, // times at which the values begin
					char * interpolations); // interpolation type codes

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
	parameter * parameters;
	int n_parameters;

	// a float array containing any state needed by this transform. Note that
	// this should be allocated at scheduling time, outside of the realtime
	// thread.
	float * state_buf;
	int state_buf_size;
	int state_buf_pos;

	// a function with the signature
	// process(in_buffer,out_buffer,struct parameter *parameters,float * state_buf)
	// E.g., For a delay, this will contain the "hold" buffer, for a convolution, the
	// impulse response
	void * process;

	// a boolean value which is true if this transform causes a pattern to have
	// an unknown length
	char unknown_length;

} transform;

typedef float (*transform_process)(jack_nframes_t time,float insample,transform * t);

// Initialize a transform
void transform_init(transform * t,
					parameter * params,
					int n_parameters,
					int state_buf_size,
					transform_process p,
					char unknown_length);

void transform_delete(transform * t);

// Gain
void gain_init(transform * t,parameter * params);
float gain_process(jack_nframes_t time,float insample,transform * gain);

// Delay
void delay_init(transform * t,int max_delay_time,parameter * params);
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
	void * children;
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

	// TODO: This is confusing, and should be named something more descriptive!
	// _done indicates that all samples have been output and/or children are
	// finished playing, but some transform may cause the event to have an
	// undetermined length
	char _done;

	// a flag indicating that one of the effects defined for this event
	// cause it to have an undetermined length. Note that if *any* "descendant"
	// nodes have an unknown length, then this node should as well.  In other
	// words, "unknown" propagates down the branches towards the root, but
	// not in the other direction.
	char unknown_length;

	// a flag for use with events who have unknown length.  This is 1 if
	// the samples have finished playing and the output has been "silent" (tbd)
	// for some tbd length of time
	int silence;

	// pointer to function which processes audio samples
	void * process;

} event2;

typedef float (*event2_process)(event2 * event,jack_nframes_t time);

// Instantiate a new raw buffer event
event2 * event2_new_buffer(
float * buf,int start_sample,int stop_sample,jack_nframes_t start_time);

void event2_new_base(
event2 * e,transform * transforms, int n_transforms,jack_nframes_t start_time_frames);

// Instantiate a new leaf event
void event2_new_leaf(
event2 * e,float * buf,int start_sample,int stop_sample,jack_nframes_t start_time,
transform * transforms,int n_transforms);

// Instantiate a new branch event
event2 * event2_new_branch(
event2 * children,int n_children,jack_nframes_t start_time,
transform * transforms,int n_transforms);

void event2_set_children(event2 * e,event2 * children,int n_events);

char event2_is_leaf(event2 * event);
char event2_is_playing(event2 * event,jack_nframes_t time);

float event2_leaf_process(event2 * event,jack_nframes_t time);
float event2_branch_process(event2 * event,jack_nframes_t time);

void event2_delete(event2 * event);


#define N_EVENTS 256
#define second_in_microsecs 1e6

// KLUDGE: This signature is all wrong

//void put_event(
//float *buf,unsigned int start_sample,unsigned int stop_sample,jack_time_t start_time_ms,char done);

void put_event(
		float * buf,
		unsigned int start_sample,
		unsigned int stop_sample,
		jack_time_t start_time_ms,
		char done);
void put_event2(event2 * e);
void cancel_all_events(void);
int render(
		jack_nframes_t nframes,
		jack_nframes_t frame_time,
		jack_default_audio_sample_t ** out,
		int channels,
		int mode);
