#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <jack/jack.h>

#include "cplay.h"

jack_client_t *client;
jack_port_t *output_ports[CHANNELS+1];
jack_port_t *input_port;


event_t *EVENTS;

/*
 * Schedule an event to be played. If the list of events is full, the event
 * will not be scheduled.
 */
void put_event(
float *buf,unsigned int start_sample,unsigned int stop_sample,jack_time_t start_time_ms,char done) {

	event_t ne;
	ne.buf = buf;
	ne.start_sample = start_sample;
	ne.stop_sample = stop_sample;
	ne.start_time_frames = jack_time_to_frames(client,start_time_ms);
	ne.done = done;
	ne.position = 0;

	int i;
	for(i = 0; i < N_EVENTS; i++) {
		if(EVENTS[i].done) {
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
	EVENTS = (event_t*)calloc(N_EVENTS,sizeof(event_t));
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
