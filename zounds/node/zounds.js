$(function() {

    function ZoundsClient() {

        this.interpret = function(command) {
            return $.ajax({
                method: 'POST',
                url: '/zounds/repl',
                data: command,
                dataType: 'json',
                contentType: 'text/plain'
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

    function Console(inputSelector, outputSelector) {
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
                        console.log(data);
                        history.push(command);
                        history_pos = 0;
                        output.append($('<div>').addClass('statement').text(command));
                    })
                    .done(function(data) {
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

    var input = new Console('#input', '#output');
});