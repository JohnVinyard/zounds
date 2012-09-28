var interval = null;

function supports_html5_storage() {
  try {
    return 'localStorage' in window && window['localStorage'] !== null;
  } catch (e) {
    return false;
  }
}

// We want the spacebar to play sounds. Keep it from paging down.
window.onkeydown=function(e){
	
	if(e.keyCode == 32) {
		return false;
	}
  
	if(e.keyCode == 82) {
		window.location.href = $('#random_search').attr('href'); 
	}
	
	if(e.keyCode == 66) {
		$('#bookmark').focus();
	}
};



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
	
	function audio_ended(a) {
		var ae = a ? a : this;
		console.log('audio ended');
		ae.pause();
		ae.currentTime = 0;
		clearInterval(interval);
		position_playhead(playhead,snippet);
	}
	
	audio.addEventListener('ended',audio_ended,false);
	
	function timeupdate() {
		clearInterval(interval);
		position_playhead(playhead,snippet);
		if(!this.paused){
			interval = setInterval(function() {
				var offset = playhead.offset();
				offset.left += increment_px;
				playhead.offset(offset);
			},update_freq_ms);
		}
	}
	
	audio.addEventListener('timeupdate',timeupdate,false);
	
	var killAudio = null;
	
	if(32 == e.keyCode) {
		if(audio.paused) {
			console.log(audio.duration);
			var cp = audio.currentPosition ? audio.currentPosition : 0;
			var ms = ((audio.duration - cp) * 1000) + 100;
			console.log(ms);
			killAudio = setTimeout(function() {
				audio_ended(audio);
				audio.removeEventListener('timeupdate',timeupdate);
			},ms);
			
			position_playhead(playhead,snippet);
			audio.play();
		}else {
			clearTimeout(killAudio);
			audio.pause();
			clearInterval(interval);
			audio.removeEventListener('timeupdate',timeupdate);
			
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


function give_query_focus() {
	$('.snippet:first').focus();
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

	// select all text when the bookmark input gets focus
	$('#bookmark').focus(function() {
		this.select();
	});
	
	// load the sound's freesound url when the attribution link is clicked
	$('.attribution').click(function() {
		var link = $(this);
		var id = link.attr('zounds_id');
		var loading_id = 'loading_' + id;
		link.replaceWith('<span id="' + loading_id + '">loading</span>');
		var loading = $('#' + loading_id);
		var interval = setInterval(function() {
			loading.text(loading.text() + '.');
			if(loading.text().indexOf('....') != -1) {
				loading.text('loading');
			}
		},500);
		
		var loading = $('#' + loading_id);
		$.get('/freesound/' + id,function(data) {
			var newlink = '<a target="_blank" href="' + data + '">' + data + '</a>';
			clearInterval(interval);
			loading.replaceWith(newlink);
		});
	});
	
	$('.ok').click(function() {
		$(this).parents('.help_text').slideToggle();
	});
	
	give_query_focus();
	$('#bookmark,.snippet:last').blur(give_query_focus);
	
	$('.toggler').click(function() {
		$('#' + $(this).attr('opens')).slideToggle();
	});
	
	
	if(supports_html5_storage()) {
		if(localStorage['about']) {
			$('#about_text').hide();
		}
		
		if(localStorage['keyboard']) {
			$('#keyboard_text').hide();
		}
		
		$('.help_text .ok').click(function() {
			var key = $(this).parents('.help_text').attr('storage_key');
			localStorage[key] = 1;
		});
	}
});