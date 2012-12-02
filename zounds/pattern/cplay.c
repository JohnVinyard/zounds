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

parameter * parameter_new(
float * values,int n_values,jack_nframes_t * times,char * interpolations) {

	struct parameter * param = malloc(sizeof(*param));

	param->n_values = n_values;
	int v_size = sizeof(float) * n_values;
	param->values = (float *)malloc(v_size);
	memmove(values,param->values,v_size);

	int t_size = sizeof(jack_nframes_t) * n_values;
	param->times = (jack_nframes_t *)malloc(t_size);
	memmove(times,param->times,t_size);

	// Number of interpolations is n - 1, since interpolations define transitions
	// between points
	int i_size = sizeof(char) * (n_values - 1);
	param->interpolations = (char *)malloc(i_size);
	memmove(interpolations,param->interpolations,i_size);

	param->pos = 0;

	return param;
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
	float current_value = parameter->values[pos];

	if(0 == interp) {
		// The interpolation type is none, so we just return the current value
		// in the values array
		return current_value;
	}

	jack_nframes_t start_time = param->times[pos];
	jack_nframes_t end_time = param->times[pos + 1];
	float next_value = param->values[pos + 1];

	if(1 == interp) {
		// linear interpolation
		// v(t) = V0 + (V1 - V0) * ((t - T0) / (T1 - T0))
		return current_value + (next_value - current_value) *
				((time - start_time) / (end_time - start_time));
	}

	// Assume that interpolation type is exponential
	// v(t) = V0 * (V1 / V0) ^ ((t - T0) / (T1 - T0))
	return current_value *
			pow((next_Value / current_value),
			(time - start_time) / (end_time - start_time));
};



void parameter_delete(parameter * param) {
	free(param->values);
	free(param->times);
	free(param->interpolations);
	free(param);
}

// Transform ##################################################################

transform * transform_new(
parameter * params,int n_parameters,int state_buf_size,void * process) {
	struct transform * t = malloc(sizeof(transform));

	// Parameters
	transform->n_parameters = n_parameters;
	transform->parameters = params;

	// State Buffer
	int s = sizeof(float) * state_buf_size;
	// allocate memory
	transform->state_buf = (float*)malloc(s);
	// zero the memory
	memset(transform->state_buf,0,s);
	transform->state_buf_size = state_buf_size;
	transform->state_buf_pos = 0;

	// Process function
	transform->process = process;
}

void transform_delete(transform * t) {
	for(int i = 0; i < t->n_parameters; i++) {
		parameter_delete(n->parameters[i]);
	}
	free(t->parameters);
	free(t->state_buf);
	free(t->process);
	free(t);
}

// Gain
float gain_process(jack_nframes_t time,float insample,transform *t) {
	float gain = parameter_current_value(t->parameters[0],time);
	return insample * gain;
}

// TODO: Gain should take a single parameter instance instead of the arguments
// to its constructor
transform * gain_new(
float * values,int n_values,jack_nframes_t * times,char * interps) {
	parameter * gain = parameter_new(values,n_values,times,interps);
	return transform_new(gain,1,0,gain_process);
}

// Delay
float delay_process(jack_nframes_t time,float insample,transform *t) {
	float effect_level = parameter_current_value(t->parameters[0],time);
	float delay_sample = t->state_buf[t->state_buf_pos];
	float out = insample + (delay_sample * effect_level);

	float feedback = parameter_current_value(t->parameters[1],time);
	t->state_buf[t->state_buf_pos] = in + (out * feedback);

	t->state_buf_pos++;
	if(t->state_buf_pos >= t->state_buf_size) {
		t->state_buf_pos = 0;
	}

	return out;
}

transform * delay_new(int max_delay_time,parameter * params) {
	return transform_new(params,3,max_delay_time,delay_process);
}

// Event ######################################################################
char event2_is_leaf(event2 * event) {
	return !event->n_children;
}

char event2_is_playing(event2 * event,jack_nframes_t time) {
	if(event_is_leaf(event)) {
		return event->position || time >= event->start_time_frames;
	}

	return time >= event->start_time_frames;
}


float event2_leaf_process(event2 * event,jack_nframes_t time) {
	// BUG: Leaf events CAN have transforms!!!

	float sample = event->buf[event->start_sample + event->position];
	event->position++;
	if(event->position > event->n_samples) {
		event->done = 1;
	}
	return sample;
}

float event2_branch_process(event2 * event,jack_nframes_t time) {

	float out = 0;
	char alive = 0;

	if(event->_done) {
		// All children are done, but this event has an unkown length because
		// of transformations defined on it or one of its children.  Keep
		// processing by feeding 0 to the processing chain.
		for(i = 0; i < event->n_transforms; i++) {
			out = event->transforms[i]->process(time,out,event->transforms[i]);
		}

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

	int i;
	for(i = 0; i < event->n_children; i++) {

		if(event->children[i]->done) {
			// The child event is done and won't contribute to the current
			// sample
			continue;
		}
		alive++;

		// TODO: event->playing property to avoid doing this comparison over
		// and over
		if(event_is_playing(event->children[i],time)) {
			out += event->children[i]->process(event->children[i],time);
		}
	}

	// Apply transformations to this sample
	for(i = 0; i < event->n_transforms; i++) {
		out = event->transforms[i]->process(time,out,event->transforms[i]);
	}


	if(!alive) {
		if(!event->unknown_length) {
			event->done = 1;
		}
		event->_done = 1;
	}

	if(!event->unknown_length && !done) {
		// This event has a predictable length, and all its children are done.
		// Mark the parent as done too.
		event->done = 1;
	}

	return out;
}

event2 * event2_new_leaf(
float * buf,int start_sample,int stop_sample,jack_nframes_t start_time) {

	// BUG: Leaf events CAN have transforms!!!

	struct event2 * event = malloc(sizeof(*event2));
	event2->buf = buf;
	event2->start_sample = start_sample;
	event2->stop_sample = stop_sample;
	event2->n_samples = stop_sample - start_sample;
	event2->position = position;

	event2->children = NULL;
	event2->n_children = 0;

	// TODO: factor this stuff out into a common constructor
	event2->n_transforms = 0;
	event2->start_time_frames = start_time;
	event2->done = 0;
	event2->_done = 0;
	event2->unknown_length = 0;
	event2->silent = 0;

	event2->process = event2_leaf_process;
}

event2 * event_new_branch(
event2 * children,int n_children,transform * transforms,jack_nframes_t start_time,
char unknown_length,transform * transforms,int n_transforms) {

	struct event2 * event = malloc(sizeof(*event2));
	event2->buf = NULL;
	event2->children = children;
	event2->n_children = n_children;

	// TODO: factor this stuff out into a common constructor
	event2->transforms = transforms;
	event2->n_transforms = n_transforms;
	event2->start_time_frames = start_time;
	event2->done = 0;
	event2->_done = 0;
	event2->unknown_length = unknown_length;
	event->silent = 0;

	event2->process = event2_branch_process;
}



void event2_delete(event2 * event) {

	// Note that event->buf isn't being freed. This is owned by the caller, and
	// may be shared between many events

	int i;
	for(i = 0; i < event->n_children; i++) {
		event2_delete(event->children[i]);
	}
	free(event->children);

	for(i = 0; i < event->n_transforms; i++) {
		transform_delete(event->transforms[i]);
	}
	free(event->transforms);
}

event2 *EVENTS;

/*
 * Schedule an event to be played. If the list of events is full, the event
 * will not be scheduled.
 */
void put_event(
float *buf,unsigned int start_sample,unsigned int stop_sample,jack_time_t start_time_ms,char done) {

	event2 ne;
	ne.buf = buf;
	ne.start_sample = start_sample;
	ne.stop_sample = stop_sample;
	ne.start_time_frames = jack_time_to_frames(client,start_time_ms);
	ne.done = done;
	ne.position = 0;

	int i;
	for(i = 0; i < N_EVENTS; i++) {
		if(EVENTS[i].done) {
			// cleanup dynamically allocated memory for the completed event
			// KLUDGE: Is this the right place to do this?
			event_delete(EVENTS[i]);
			EVENTS[i] = ne;
			break;
		}
	}
}

jack_time_t get_time(void) {
	jack_nframes_t frames = jack_frame_time(client);
	return jack_frames_to_time(client,frames);
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

			/*
			if(EVENTS[j].position > 0 || frame_time >= EVENTS[j].start_time_frames) {
				// the sample has already started, or starts on this frame
				current_pos = EVENTS[j].start_sample + EVENTS[j].position;
				sample+= EVENTS[j].buf[current_pos];
				// advance the sample position of this event
				EVENTS[j].position++;

				if((current_pos + 1) >= EVENTS[j].stop_sample) {
					// Mark this event as done
					EVENTS[j].done = 1;
				}
			}
			*/
			if(event2_is_playing(EVENTS[j])) {
				sample += EVENTS[j].process(EVENTS[j],frame_time);
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
