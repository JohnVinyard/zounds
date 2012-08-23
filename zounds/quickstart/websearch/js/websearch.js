var interval = null;

window.onkeydown=function(e){
  if(e.keyCode==32){
   return false;
  }
};

function focus_query() {
	$('.snippet:first').focus();
}

function position_playhead(playhead,snippet) {
	var a = snippet.find('audio')[0];
	var img = snippet.find('.spectrogram');
	var offset = img.offset();
	var width = img.width();
	var ratio = a.currentTime / a.duration;
	offset.left += width * ratio
	playhead.show().offset(offset);
}

function audio_spacebar(e) {
	var snippet = $(this);
	var audio = snippet.find('audio')[0];
	var img = snippet.find('.spectrogram');
	var playhead = $('#playhead');
	var pps = img.width() / audio.duration;
	var update_freq_ms = 50;
	var increment_px = pps * (update_freq_ms / 1000);
	
	audio.addEventListener('ended',function() {
		this.pause();
		this.currentTime = 0;
		clearInterval(interval);
		position_playhead(playhead,snippet);
	},false);
	
	audio.addEventListener('timeupdate',function() {
		clearInterval(interval);
		position_playhead(playhead,snippet);
		if(!this.paused){
			interval = setInterval(function() {
				var offset = playhead.offset();
				offset.left += increment_px;
				playhead.offset(offset);
			},update_freq_ms);
		}
	},false);
	
	if(32 == e.keyCode) {
		if(audio.paused) {
			position_playhead(playhead,snippet);
			audio.play();
		}else {
			audio.pause();
			clearInterval(interval);
			audio.removeEventListener('timeupdate');
		}
		return false;
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
		var snippet = $(this);
		position_playhead(playhead,snippet);
		snippet
			.addClass('focus')
			.keyup(audio_spacebar);
		var img = snippet.find('.spectrogram');
		img.click(function(e) {
			var offset = playhead.offset();
			offset.left = e.pageX;
			var img_offset = $(this).offset();
			var ratio = (e.pageX - img_offset.left) / $(this).width();
			var audio = snippet.find('audio')[0];
			audio.currentTime = ratio * audio.duration;
			position_playhead(playhead,snippet);
		});
	});
	
	$('.snippet').blur(function() {
		$(this).find('audio')[0].pause();
		clearInterval(null);
		$(this)
			.removeClass('focus')
			.unbind('keyup');
		var img = $(this).find('.spectrogram');
		img.unbind('click');
	});
});