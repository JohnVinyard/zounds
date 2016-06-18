function ZoundsClient(endpoints) {

    var etags = {};

    // add custom endpoints
    if(endpoints) {
        for(var key in endpoints) {
            if(!endpoints.hasOwnProperty(key)) { continue; }
            this[key] = function(data) {
                return $.ajax(endpoints[key](data));
            };
        }
    }


    this.fetchBinary = function(url, range) {
        return new Promise(function(resolve, reject) {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url);
            if(range) {
                xhr.setRequestHeader(
                    'Range',
                    'seconds=' + range.start.toString() + '-' + range.stop.toString());
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