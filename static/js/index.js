// Scene switching
function showScene(containerId, sceneId, btn) {
    var container = document.getElementById(containerId);
    var scenes = container.querySelectorAll('.scene-content');
    for (var i = 0; i < scenes.length; i++) {
        scenes[i].classList.remove('active');
        var vids = scenes[i].querySelectorAll('video');
        for (var j = 0; j < vids.length; j++) vids[j].pause();
    }

    document.getElementById(sceneId).classList.add('active');

    var allButtons = document.querySelectorAll('#' + containerId + '-selector .scene-button');
    for (var i = 0; i < allButtons.length; i++) {
        allButtons[i].classList.remove('active');
    }
    btn.classList.add('active');

    // Reset and play all videos together
    var activeScene = document.getElementById(sceneId);
    var videos = activeScene.querySelectorAll('video');
    for (var i = 0; i < videos.length; i++) {
        videos[i].currentTime = 0;
        videos[i].play();
    }
}

// On page load, ensure all visible videos play
document.addEventListener('DOMContentLoaded', function() {
    // Play all videos in active scenes
    // Must be muted for browsers (Safari/iOS) that block autoplay of videos
    // with audio. Catch NotAllowedError rejection so console stays clean.
    var activeScenes = document.querySelectorAll('.scene-content.active');
    for (var i = 0; i < activeScenes.length; i++) {
        var videos = activeScenes[i].querySelectorAll('video');
        for (var j = 0; j < videos.length; j++) {
            videos[j].muted = true;
            var p = videos[j].play();
            if (p && typeof p.catch === 'function') p.catch(function(){});
        }
    }

    // Simple sync: when any video loops back to start, reset all siblings
    document.querySelectorAll('video').forEach(function(video) {
        video.addEventListener('ended', function() {
            // This won't fire with loop, but just in case
        });
    });

    // Light periodic sync - just nudge, never pause
    setInterval(function() {
        var activeScenes = document.querySelectorAll('.scene-content.active');
        activeScenes.forEach(function(scene) {
            var videos = scene.querySelectorAll('video');
            if (videos.length < 2) return;
            var target = videos[0].currentTime;
            for (var i = 1; i < videos.length; i++) {
                if (Math.abs(videos[i].currentTime - target) > 0.3) {
                    videos[i].currentTime = target;
                }
            }
        });
    }, 2000);

    if (typeof bulmaCarousel !== 'undefined') {
        bulmaCarousel.attach('#results-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            autoplay: true,
            autoplaySpeed: 5000,
        });
    }
});
