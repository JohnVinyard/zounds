#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <jack/jack.h>

#include "cplay.h"

jack_client_t *client;
jack_port_t *output_ports[CHANNELS+1];
jack_port_t *input_port;

// Parameter ##################################################################

void parameter_init(
parameter * param,float * values,int n_values,jack_nframes_t * times,char * interpolations) {

	// values are copied since these values may come from numpy arrays which are
	// garbage collected if a reference to them doesn't exist in Python space.
	param->n_values = n_values;
	int v_size = sizeof(float) * n_values;
	param->values = (float *)malloc(v_size);
	memmove(param->values,values,v_size);

	int t_size = sizeof(jack_nframes_t) * n_values;
	param->times = (jack_nframes_t *)malloc(t_size);
	memmove(param->times,times,t_size);


	// Number of interpolations is n - 1, since interpolations define transitions
	// between points
	int i_size = sizeof(char) * (n_values - 1);
	param->interpolations = (char *)malloc(i_size);
	memmove(param->interpolations,interpolations,i_size);

	param->pos = 0;
};

int parameter_advance_if_necessary(parameter * param,jack_nframes_t time) {
	int pos = param->pos;

	if(pos == param->n_values - 1) {
		// -1 signifies that the last position has been reached
		return -1;
	}

	jack_nframes_t next_time = param->times[pos + 1];
	if(time >= next_time) {
		param->pos++;
	}
	return param->pos;
};

float parameter_current_value(parameter * param,jack_nframes_t time) {

	// check if it's time to move on to the next position
	int pos = parameter_advance_if_necessary(param,time);

	if(-1 == pos) {
		// -1 signifies that the last value in the values array has been
		// reached.  We'll return this value forever.
		return param->values[param->pos];
	}

	char interp = param->interpolations[pos];
	float current_value = param->values[pos];

	if(0 == (int)interp) {
		// The interpolation type is none, so we just return the current value
		// in the values array
		return current_value;
	}

	jack_nframes_t start_time = param->times[pos];
	jack_nframes_t end_time = param->times[pos + 1];
	float next_value = param->values[pos + 1];

	float rel_time = (float)(time - start_time) / (float)(end_time - start_time);
	if(1 == (int)interp) {
		// linear interpolation
		// v(t) = V0 + (V1 - V0) * ((t - T0) / (T1 - T0))
		return current_value + ((next_value - current_value) * rel_time);
	}

	// Assume that interpolation type is exponential
	// v(t) = V0 * (V1 / V0) ^ ((t - T0) / (T1 - T0))
	return current_value * pow((next_value / current_value),rel_time);
};



void parameter_delete(parameter * param) {
	free(param->values);
	free(param->times);
	free(param->interpolations);
}

// Transform ##################################################################

void transform_init(
transform * t,parameter * params,int n_parameters,
int state_buf_size,transform_process p,char unknown_length) {
	// Parameters
	t->n_parameters = n_parameters;
	t->parameters = params;

	// State Buffer
	int s = sizeof(float) * state_buf_size;
	// allocate memory
	t->state_buf = (float*)malloc(s);
	// zero the memory
	memset(t->state_buf,0,s);
	t->state_buf_size = state_buf_size;
	t->state_buf_pos = 0;

	// Process function
	t->process = (void*)p;

	t->unknown_length = unknown_length;
}

void transform_delete(transform * t) {
	int i;
	for(i = 0; i < t->n_parameters; i++) {
		parameter_delete(&(t->parameters[i]));
	}
	free(t->parameters);
	free(t->state_buf);
}

// Gain
float gain_process(jack_nframes_t time,float insample,transform *t) {
	float gain = parameter_current_value(&(t->parameters[0]),time);
	return insample * gain;
}

void gain_init(transform * t,parameter * params) {
	transform_init(
			t,
			params,       // transform parameters
			1,            // number of parameters
			0,            // size of the state buffer
			gain_process, // process function pointer
			0);           // known length
}

// Delay
// TODO: How do I avoid clicks when then delay time is changed?
float delay_process(jack_nframes_t time,float insample,transform *t) {

	float effect_level = parameter_current_value(&(t->parameters[0]),time);
	float delay_sample = t->state_buf[t->state_buf_pos];
	float out = insample + (delay_sample * effect_level);

	float feedback = parameter_current_value(&(t->parameters[1]),time);
	t->state_buf[t->state_buf_pos] = insample + (out * feedback);

	t->state_buf_pos++;
	// KLUDGE: What about other sample rates?
	int delay_time_samples =
			(int)(parameter_current_value(&(t->parameters[2]),time) * 44100.);
	if(t->state_buf_pos >= delay_time_samples) {
		t->state_buf_pos = 0;
	}

	return out;
}

void delay_init(transform * t,int max_delay_time,parameter * params) {
	return transform_init(
			t,
			params,         // transform parameters
			3,              // number of parameters
			max_delay_time, // size of the state buffer
			delay_process,  // process function pointer
			1);             // unknown length
}

// Event ######################################################################
char event2_is_leaf(event2 * event) {
	return !event->n_children;
}

char event2_is_playing(event2 * event,jack_nframes_t time) {
	if(event2_is_leaf(event)) {
		return event->position || time >= event->start_time_frames;
	}
	return time >= event->start_time_frames;
}

float event2_do_transforms(event2 * event, float insample,jack_nframes_t time) {
	int i;
	float out = insample;
	// Apply transformations to this sample
	for(i = 0; i < event->n_transforms; i++) {
		transform_process p = (transform_process)(event->transforms[i].process);
		out = p(time,out,&(event->transforms[i]));
	}
	return out;
}

float event2_set_done(event2 * event) {
	if(!event->unknown_length) {
		event->done = 1;
	}
	event->_done = 1;
}

float event2_do_tail(event2 * event,jack_nframes_t time) {
	float out = 0;
	out = event2_do_transforms(event,out,time);

	if(out < SILENCE_THRESHOLD) {
		event->silence++;
	}else {
		event->silence = 0;
	}

	if(event->silence >= KILL_AFTER) {
		// This event has been outputting "silence" for some amount of time.
		// Mark it as done.
		event->done = 1;
	}

	return out;
}

// TODO: Times aren't being created / handled relative to parent patterns!
float event2_leaf_process(event2 * event,jack_nframes_t time) {

	if(event->_done) {
		return event2_do_tail(event,time);
	}

	// read from the buffer
	float sample = event->buf[event->start_sample + event->position];
	// do transformations
	// TODO: Should I be passing a relative time to the transform?
	sample = event2_do_transforms(event,sample,event->position);
	// advance the buffer position
	event->position++;

	if(event->position >= event->n_samples) {
		event2_set_done(event);
	}

	return sample;
}

// TODO: Times aren't being created / handled relative to parent patterns!
float event2_branch_process(event2 * event,jack_nframes_t time) {
	if(event->_done) {
		return event2_do_tail(event,time);
	}

	float out = 0;
	char alive = 0;
	int i;
	event2 * children = (event2*)(event->children);
	jack_nframes_t rel_time = time - event->start_time_frames;

	for(i = 0; i < event->n_children; i++) {
		if(children[i].done) {
			// The child event is done and won't contribute to the current
			// sample
			continue;
		}
		alive++;

		// TODO: event->playing property to avoid doing this comparison over
		// and over
		if(event2_is_playing(&(children[i]),rel_time)) {
			event2_process p = (event2_process)(children[i].process);
			out += p(&(children[i]),rel_time);
		}
	}

	// Apply transformations to this sample
	out = event2_do_transforms(event,out,rel_time);

	if(!alive) {
		event2_set_done(event);
	}

	return out;
}

void event2_new_common(
event2 * e,transform * transforms, int n_transforms,
jack_nframes_t start_time_frames) {

	// determine if any transforms defined on this pattern cause it to have
	// an unknown length
	char unknown_length = 0;
	int i = 0;
	for(i = 0; i < n_transforms; i++) {
		if(transforms[i].unknown_length) {
			unknown_length = 1;
			break;
		}
	}

	e->transforms = transforms;
	e->n_transforms = n_transforms;
	e->start_time_frames = start_time_frames;
	e->done = 0;
	e->_done = 0;
	e->unknown_length = unknown_length;
	e->silence = 0;
}

void event2_new_base(
event2 * e,transform * transforms,int n_transforms,jack_nframes_t start_time_frames) {
	e->process = (void*)event2_branch_process;
	e->children = NULL;
	e->n_children = 0;
	event2_new_common(e,transforms,n_transforms,start_time_frames);
}

event2 * event2_new_buffer(
float * buf,int start_sample,int stop_sample,jack_nframes_t start_time) {

	event2 * e = (event2*)malloc(sizeof(event2));
	transform * t = (transform*)malloc(0);
	event2_new_leaf(
			e,
			buf,
			start_sample,
			stop_sample,
			start_time,
			t,           // empty transform array
			0);          // zero transforms

	return e;
}

void event2_new_leaf(
event2 * e,float * buf,int start_sample,int stop_sample,jack_nframes_t start_time,
transform * transforms,int n_transforms) {

	e->buf = buf;
	e->start_sample = start_sample;
	e->stop_sample = stop_sample;
	e->n_samples = stop_sample - start_sample;
	e->position = 0;

	e->children = NULL;
	e->n_children = 0;

	event2_new_common(
			e,transforms,n_transforms,start_time);

	e->process = (void*)event2_leaf_process;
}



// TODO: Times aren't being created / handled relative to parent patterns!
event2 * event_new_branch(
event2 * children,int n_children,jack_nframes_t start_time,
transform * transforms,int n_transforms) {

	event2 * e = malloc(sizeof(event2));

	event2_new_common(
				e,transforms,n_transforms,start_time);

	e->buf = NULL;
	// TODO: use the event2_set_children method
	event2_set_children(e,children,n_children);

	e->process = (void*)event2_branch_process;

	return e;
}

void event2_set_children(event2 * e,event2 * children,int n_children) {
	e->children = (void*)children;
	e->n_children = n_children;

	int i;
	for(i = 0; i < n_children; i++) {
		if(children[i].unknown_length) {
			e->unknown_length = 1;
			break;
		}
	}
}


event2 * event2_new_branch_incomplete(
jack_nframes_t start_time,transform * transforms,int n_transforms) {

	event2 * e = malloc(sizeof(event2));
	e->buf = NULL;

}


void event2_delete(event2 * event) {
	// Note that event->buf isn't being freed. This is owned by the caller, and
	// may be shared between many events
	int i;
	event2 * children = (event2*)event->children;
	for(i = 0; i < event->n_children; i++) {
		event2_delete(&(children[i]));
	}
	free(event->children);

	for(i = 0; i < event->n_transforms; i++) {
		transform_delete(&(event->transforms[i]));
	}
	free(event->transforms);
}

event2 * EVENTS;

void put_event2(event2 * e) {
	int i;
	for(i = 0; i < N_EVENTS; i++) {
		if(EVENTS[i].done) {
			// cleanup dynamically allocated memory for the completed event
			// KLUDGE: Is this the right place to do this?
			event2_delete(&(EVENTS[i]));
			EVENTS[i] = *e;
			break;
		}
	}
}

void put_event(
float * buf,unsigned int start_sample,unsigned int stop_sample,
jack_time_t start_time_ms,char done) {
	event2 * leaf = (event2*)malloc(sizeof(event2));
	transform * t = (transform*)malloc(0);
	event2_new_leaf(
			leaf,
			buf,
			start_sample,
			stop_sample,
			jack_time_to_frames(client,start_time_ms),
			t,
			0);
	put_event2(leaf);
}

jack_time_t get_time(void) {
	jack_nframes_t frames = jack_frame_time(client);
	return jack_frames_to_time(client,frames);
}

jack_nframes_t get_frame_time(void) {
	return jack_frame_time(client);
}

void cancel_all_events(void) {
	int i;
	for (i = 0; i < N_EVENTS; i++) {
		EVENTS[i].done = 1;
	}
}

/*
 * Initialize all events
 */
void init_events(void) {
	EVENTS = (event2*)calloc(N_EVENTS,sizeof(event2));
	cancel_all_events();
}



int process(jack_nframes_t nframes, void *arg) {
	jack_default_audio_sample_t *out[CHANNELS+1];
	jack_default_audio_sample_t *in;

	int i,j,k,current_pos;
	for (i = 0; i < CHANNELS; ++i) {
		out[i] = jack_port_get_buffer(output_ports[i], nframes);
	}

	jack_nframes_t frame_time = jack_last_frame_time(client);
	float sample = 0;


	for(i = 0; i < nframes; i++) {
		sample = 0;

		// KLUDGE: This is a lot of unecessary work. I should figure out
		// which events are playing, or will begin in this cycle, and only
		// loop over those.
		for(j = 0; j < N_EVENTS; j++) {


			if(EVENTS[j].done) {
				// this position hasn't been assigned yet, or the sample has
				// finished playing
				continue;
			}

			event2 * e = &(EVENTS[j]);
			if(event2_is_playing(e,frame_time)) {
				event2_process p = (event2_process)(e->process);
				sample += p(e,frame_time);
			}
		}

		// Write the computed sample to all channels of the out buffer
		for(k = 0; k < CHANNELS; k++) {
			out[k][i] = sample;
		}

		// advance the frame time by one
		frame_time++;
	}

	return 0;
}



void jack_shutdown(void *arg) {
	exit(1);
}

jack_client_t *jack_start(void) {
	const char **ports;
	const char *client_name = "zounds";
	const char *server_name = NULL;
	jack_options_t options = JackNullOption;
	jack_status_t status;
	int i;
	char portname[16];


	//open a client connection to the JACK server

	client = jack_client_open(client_name, options, &status, server_name);
	if (client == NULL) {
		fprintf(stderr, "jack_client_open() failed, "
				"status = 0x%2.0x\n", status);
		if (status & JackServerFailed) {
			fprintf(stderr, "Unable to connect to JACK server\n");
		}
		exit(1);
	}
	if (status & JackServerStarted) {
		fprintf(stderr, "JACK server started\n");
	}
	if (status & JackNameNotUnique) {
		client_name = jack_get_client_name(client);
		fprintf(stderr, "unique name `%s' assigned\n", client_name);
	}

	jack_set_process_callback(client, process, NULL);

	jack_on_shutdown(client, jack_shutdown, 0);

	printf("engine sample rate: %" PRIu32 "\n",
			jack_get_sample_rate(client));

	strcpy(portname, "input");
	input_port = jack_port_register(client, portname,
			JACK_DEFAULT_AUDIO_TYPE,
			JackPortIsInput, 0);

	if (input_port == NULL) {
		fprintf(stderr, "no JACK input ports available\n");
		exit(1);
	}

	for (i = 0; i < CHANNELS; ++i) {
		sprintf(portname, "output_%d", i);
		output_ports[i] = jack_port_register(client, portname,
				JACK_DEFAULT_AUDIO_TYPE,
				JackPortIsOutput, 0);
		if (output_ports[i] == NULL) {
			fprintf(stderr, "no more JACK ports available\n");
			exit(1);
		}
	}

	output_ports[CHANNELS] = NULL;

	if (jack_activate(client)) {
		fprintf(stderr, "cannot activate client");
		exit(1);
	}

	ports = jack_get_ports(client, NULL, NULL,
			JackPortIsPhysical|JackPortIsInput);
	for (i = 0; i < CHANNELS; ++i) {
		if (ports[i] == NULL) {
			break;
		}

		if (jack_connect(client, jack_port_name(output_ports[i]), ports[i])) {
			fprintf(stderr, "cannot connect output ports\n");
		}
	}

	ports = jack_get_ports(client, NULL, NULL,
			JackPortIsPhysical|JackPortIsOutput);

	if (jack_connect(client, ports[0], jack_port_name(input_port))) {
		fprintf(stderr, "cannot connect input port\n");
	}

	free(ports);

	return(client);
}


void setup(void) {
	init_events();
	jack_start();
}

void teardown(void) {
	jack_client_close(client);
	free(EVENTS);
}
