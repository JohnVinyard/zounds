function AudioSlice(data, root, context, client) {
    var self = this;

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