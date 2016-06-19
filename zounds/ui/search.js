$(function() {

    var client = new ZoundsClient({
        randomSearch : function() {
            return {
                method: 'GET',
                url: '/zounds/search',
                dataType: 'json'
            };
        }
    });

    var audioContext = new AudioContext();

    function showResults() {
        var root = $('#results');
        client.randomSearch().then(function(resp) {
            root.empty();
            console.log(resp.results);
            for(var i = 0; i < resp.results.length; i++) {
                new AudioSlice(resp.results[i], root, audioContext, client);
            }
        });
    }

    $('#random-search').click(showResults);

    showResults();
});