$(function() {
    var url = 'ws://' + window.location.host + '/zounds/training';
    var ws = new WebSocket(url);
    ws.onmessage = function (evt) {
       data = JSON.parse(evt.data);
       console.log(data);
       $('#training-monitor').text(JSON.stringify(data));
    };

    window.setInterval(function(){
        $('#graph img').attr('src', '/zounds/graph?q=' + Math.random());
    }, 5000);
});