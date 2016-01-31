$(function() {

    var events = {
        FEATURE_RECEIVED: 'FEATURE_RECEIVED'
    };

    function MessageBus() {

        this.publish = function(event, data) {
            $(document).trigger(event, data);
        }

        this.subscribe = function(event, func) {
            $(document).on(event, func);
        }

    }

    function AudioSlice(data, root, context) {
        var
            self = this,
            client = new ZoundsClient();

        var
            slice = {
                start: data.slice.start,
                stop: data.slice.start + data.slice.duration
            },
            audio = client.fetchAudio(data.audio, slice, context),
            vis = client.fetchBinary(data.visualization, slice),
            aggregate = Promise.all([audio, vis]);

        this.play = function() {
            var source = context.createBufferSource();
            source.buffer = self.audio;
            source.connect(context.destination);
            source.start(0);
        }

        function arrayBufferToBase64(buffer) {
            var binary = '';
            var bytes = new Uint8Array(buffer);
            var len = bytes.byteLength;
            for (var i = 0; i < len; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            return window.btoa(binary);
        }

        aggregate.then(function(results) {
            var
                audio = results[0],
                image = results[1];

            self.audio = audio;
            $('<img>')
                .attr('src', 'data:image/png;base64,' + arrayBufferToBase64(image))
                .appendTo(root)
                .click(function() {
                    self.play();
                });
        });
    }

    function SearchResults(data, root, context) {
        var
            position = 0,
            container = $('<div>'),
            el = $('<div>'),
            self = this;

        this.render = function() {
            container.empty();
            new AudioSlice(data[position], container, context);
        }

        this.next = function() {
            position += 1;
            if(position >= data.length){
                position = 0;
            }
            self.render();
        }

        this.previous = function() {
            position -= 1;
            if(position < 0) {
                position = data.length - 1;
            }
            self.render();
        }

        el.append(container);
        $('<a href="javascript:void(0);">previous</a>')
            .appendTo(el)
            .click(function() {
                self.previous();
            });
        $('<a href="javascript:void(0);">next</a>')
            .appendTo(el)
            .click(function() {
                self.next();
            });
        root.append(el);
        self.render();
    }

    function Visualization(selector, bus, context) {
        var
            el = $(selector),
            client = new ZoundsClient();

        bus.subscribe(events.FEATURE_RECEIVED, function(event, data) {
            el.empty();

            if(data.contentType === 'image/png') {
                $('<img>').attr('src', data.url).appendTo(el);
                return;
            }

            if(data.contentType === 'audio/ogg') {
                var audio = $('<audio>').attr('controls', true);
                $('<source>').attr('src', data.url).appendTo(audio);
                audio.appendTo(el);
                return;
            }

            if(data.contentType == 'application/vnd.zounds.searchresults+json') {
                $.ajax({
                    method: 'GET',
                    url: data.url,
                    dataType: 'json'
                }).done(function(resp) {
                    new SearchResults(resp.results, el, context);
                });
            }
        });

    }

    function ZoundsClient() {

        this.interpret = function(command) {
            return $.ajax({
                method: 'POST',
                url: '/zounds/repl',
                data: command,
                dataType: 'json',
                contentType: 'text/plain'
            });
        };

        this.fetchBinary = function(url, range) {
            return new Promise(function(resolve, reject) {
                var xhr = new XMLHttpRequest();
                xhr.open('GET', url);
                if(range) {
                    xhr.setRequestHeader('Range', 'seconds=' + range.start.toString() + '-' + range.stop.toString());
                }
                xhr.responseType = 'arraybuffer';
                xhr.onload = function() {
                    if(this.status >= 200 && this.status < 300) {
                        resolve(xhr.response);
                    } else {
                        reject(this.status, xhr.statusText);
                    }
                };
                xhr.onerror = function() {
                    reject(this.status, xhr.statusText);
                };
                xhr.send();
            });
        };

        this.fetchAudio = function(url, range, context) {
            return this
                .fetchBinary(url, range)
                .then(function(data) {
                    return new Promise(function(resolve, reject) {
                        context.decodeAudioData(data, function(buffer) {
                            resolve(buffer);
                        });
                    });
                });
        }
    }

    function History() {

        var
            history = [],
            hash = {};

        this.push = function(item) {
            if(!item || item === '') { return; }
            if(item in hash) { return; }
            history.push(item);
            hash[item] = true;
        }

        this.fetch = function(index) {
            var value = history[history.length - index];
            return value;
        }

        this.count = function() {
            return history.length;
        }
    }

    function Console(inputSelector, outputSelector, messageBus) {
        var
            input = $(inputSelector),
            output = $(outputSelector),
            history = new History(),
            history_pos = 0,
            client = new ZoundsClient();

        function fetch() {
            value = history.fetch(history_pos);
            input.val(value);
            if(!value) { return; }
            setTimeout(function() {
                input.get(0).setSelectionRange(value.length, value.length);
            }, 0);
        }

        $(document).keydown(function(e) {

            if(e.which === 38 && history_pos < history.count()) {
                history_pos += 1;
                fetch();
                return;
            }

            if(e.which === 40 && history_pos > 0) {
                history_pos -= 1;
                fetch();
                return;
            }
        });

        input.keypress(function(e) {
            var command = $(this).val();

            if(e.which === 13) {
                client
                    .interpret(command)
                    .always(function(data) {
                        history.push(command);
                        history_pos = 0;
                        output.append($('<div>').addClass('statement').text(command));
                    })
                    .done(function(data) {
                        if(data.url && data.contentType) {
                            messageBus.publish(events.FEATURE_RECEIVED, data);
                        }
                        output.append($('<div>').addClass('result').text(data.result));
                        window.scrollTo(0,document.body.scrollHeight);
                    })
                    .fail(function(data) {
                        output.append($('<div>').addClass('error').text(data.responseJSON.error));
                        window.scrollTo(0 ,document.body.scrollHeight);
                    });
                $(this).val('');
            }
        });
    }

    var audioContext = new AudioContext();
    var bus = new MessageBus();
    var input = new Console('#input', '#output', bus);
    var vis = new Visualization('#visualization', bus, audioContext);
});