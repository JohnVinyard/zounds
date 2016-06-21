$(function() {

    var client = new ZoundsClient({
        randomSearch : function(query) {
            return {
                method: 'GET',
                url: '/zounds/search',
                data: query ? { query: query } : null,
                dataType: 'json'
            };
        }
    });

    var audioContext = new AudioContext();

    function showResults(honorQueryString) {
        var root = $('#results');
        var query = honorQueryString ?
            window.location.search.substring(1).split('query=')[1]
            : null;
        client.randomSearch(query).then(function(resp) {
            root.empty();
            for(var i = 0; i < resp.results.length; i++) {
                new AudioSlice(resp.results[i], root, audioContext, client);
            }
            var newUrl = '/?query=' + encodeURIComponent(resp.query);
            history.pushState({}, '', newUrl);
        });
    }

    $('#random-search').click(function() { showResults(false); });

    showResults(true);
});