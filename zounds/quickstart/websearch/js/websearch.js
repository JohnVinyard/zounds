function focus_query() {
	$('.snippet:first').focus();
}

function position_playhead(playhead,snippet) {
	var a = snippet.find('audio')[0];
	var img = snippet.find('img');
	var offset = img.offset();
	var width = img.width();
	var ratio = a.currentTime / a.duration;
	offset.left += width * ratio
	playhead.show().offset(offset);
}

function audio_spacebar(e) {
	var snippet = $(this);
	var audio = snippet.find('audio')[0];
	var img = snippet.find('img');
	var playhead = $('#playhead');
	
	audio.addEventListener('ended',function() {
		console.debug('audio ended');
		this.pause();
		this.currentTime = 0;
		position_playhead(playhead,snippet);
	},false);
	
	if(32 == e.keyCode) {
		if(audio.paused) {
			position_playhead(playhead,snippet);
			var ratio =  audio.currentTime / audio.duration;
			playhead.animate({
				left : img.offset().left + img.width()
			},(audio.duration - audio.currentTime) * 1000,'linear');
			audio.play();
		}else {
			audio.pause();
			playhead.stop();
		}
	}
}

function draw_playhead(jqp) {
	var p = jqp[0];
	ctxt = p.getContext('2d');
	ctxt.fillStyle = '#00ff00';
	ctxt.fillRect(0,0,2,200);
}


$(function() {
	var playhead = $('#playhead');
	draw_playhead(playhead);
	
	$('.snippet').focus(function() {
		position_playhead(playhead,$(this));
		$(this)
			.addClass('focus')
			.keyup(audio_spacebar);
	});
	
	$('.snippet').blur(function() {
		$(this)
			.removeClass('focus')
			.unbind('keyup');
	});
});